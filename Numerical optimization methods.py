import math
from tkinter import *
import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy import optimize

from scipy.optimize import minimize


def evklideNorm(tList):
    evNorm =0
    for i in range (len(tList)):
        evNorm += tList[i]*tList[i]
    evNorm = math.sqrt(evNorm)
    return evNorm

def newQuzixMatrix(H,X,GradF,k):
    s = []
    for i in range(len(X)):
        s.append(X[i][k]-X[i][k-1])
    y =[Gradi-Gradk for Gradi,Gradk in zip(GradF[k],GradF[k-1])]
    vecY = np.matrix(y)
    vecS = np.matrix(s)
    vectY = vecY.transpose()
    H.append(H[k-1]- (H[k-1]*vectY*vecY*H[k-1])/(vecY*H[k-1]*vectY) + (vecS.transpose()*vecS)/(vecS*vectY))

def summList(list1, list2):
        newList = []
        if (len(list2) >= len(list1)):
            for i in range(len(list1)):
                newList.append(list1[i] + list2[i])
        return newList

def proizvListNum(list, num):
        newList = []
        for i in range(len(list)):
            newList.append(list[i] * num)
        return newList





class App(tk.Tk):
    def __init__(self):
        super(App, self).__init__()

        self.title("Лабораторная работа №1")
        self.minsize(600, 400)

        tabControl = ttk.Notebook(self)
        tab1 = ttk.Frame(tabControl)
        tabControl.add(tab1, text="Одномерная оптимизация")

        # Выбор функции
        r_var = IntVar()
        r_var.set(0)
        test_func = Radiobutton(tab1, text="(x-1)^2", font=("Georgia", 12), variable=r_var, value=0)
        test_func.grid(column=0, row=1, sticky = W)

        func_1_2_1_a = Radiobutton(tab1, text="4x^3-8x^2-11x+5", font=("Georgia", 12), variable=r_var, value=1)
        func_1_2_1_a.grid(column=0, row=2, sticky = W)

        func_1_2_1_b = Radiobutton(tab1, text="x+3/x^2", font=("Georgia", 12), variable=r_var, value=2)
        func_1_2_1_b.grid(column=0, row=3, sticky = W)
        #
        lbl_f_x = Label(tab1, text="Функция", font=("Georgia", 12), pady=1, padx=1)
        lbl_f_x.grid(column=0, row=0, sticky = W)
        #
        lbl_x_0 = Label(tab1, text="x^0", font=("Georgia", 12))
        lbl_x_0.grid(column=0, row=4)
        str_x_0 = Entry(tab1, width=20)
        str_x_0.grid(column=1, row=4)
        #
        lbl_h = Label(tab1, text="шаг h", font=("Georgia", 12))
        lbl_h.grid(column=0, row=5)
        str_h = Entry(tab1, width=20)
        str_h.grid(column=1, row=5)
        #
        sresult = Label(tab1, text="Результат:", font=("Georgia", 12))
        sresult.grid(column=0, row=8)
        showed_result = Label(tab1, text='0',
                              fg='Black', font='Arial 14')
        showed_result.grid(column = 1, row = 8)

        def test(x):
            if r_var.get() == 0:
                return np.power((x-1),2)
            elif r_var.get() == 1:
                return 4*x**3-8*x**2-11*x+5
            elif r_var.get() == 2:
                return x+(3/(x**2))


        def search_local_min():
            x0 = str_x_0.get()
            x0 = float(x0)
            h = str_h.get()
            h = float(h)

            f0 = test(x0)
            a = b = x0
            if f0 > test(x0 + h):
                a = x0


            elif test(x0 - h) >= f0:
                a = x0 - h
                b = x0 + h
                showed_result['text'] = "[", str(a), "", str (b), "]"
            else:
                b = x0
                h = -h

            def xk(k: int):
                return x0 + (2 ** (k - 1)) * h

            def assign_if(is_a, value):
                nonlocal a, b
                if is_a:
                    a = value
                else:
                    b = value

            k = 2
            while True:
                xk0, xk1 = xk(k), xk(k - 1)
                if test(xk0) >= test(xk1):
                    assign_if(h < 0, xk0)
                    break
                else:
                    assign_if(h > 0, xk1)
                k += 1

            showed_result['text'] = "[", str(a), "", str (b), "]"


        calculate = Button(tab1, text="Поиск отрезка", command=search_local_min)
        calculate.grid(column=1, row=7)

        passiv_poisk = Label(tab1, text = "Метод пассивного поиска",  font=("Georgia", 12))
        passiv_poisk.grid(column = 3, row = 0)

        lbl_x_okrest = Label(tab1, text="Введите x", font=("Georgia", 12))
        lbl_x_okrest.grid(column=3, row=1)
        str_x_okrest = Entry(tab1, width=20)
        str_x_okrest.grid(column=4, row=1)

        sresult_pass = Label(tab1, text="Результат:", font=("Georgia", 12))
        sresult_pass.grid(column=3, row=3)
        showed_result_pass = Label(tab1, text='0',
                              fg='Black', font='Arial 14')
        showed_result_pass.grid(column=4, row=3)

        def passive_search():

            N = 20
            h = str_x_okrest.get()
            h = float(h)
            a_g = h-2
            b_g = h+2

            step = (b_g - a_g) / N
            x_points = np.arange(a_g + step, b_g + step, step)
            print(x_points)
            value_points = np.array([test(x) for x in x_points])
            print(value_points)
            min_index = value_points.argmin()
            print(min_index)
            min_value = test(x_points[min_index])
            print(min_value)
            showed_result_pass['text'] = "x*=", str(x_points[min_index]), "f*=",  str(min_value)


        calculate_pass = Button(tab1, text="Поиск точки", command=passive_search)
        calculate_pass.grid(column=4, row=2)

# 2 Вкладка
################################################
        tab2 = ttk.Frame(tabControl)
        tabControl.add(tab2, text="Многомерная оптимизация")
        tabControl.pack(expan=1, fill="both")
        nameMethod = Label(tab2, text="Метод конфигураций Хука-Дживса", font=("Georgia", 12))
        nameMethod.grid(column=0, row=4)

        lbl_start_p = Label(tab2, text="Начальные точки х0 =", font=("Georgia", 12))
        lbl_start_p.grid(column=0, row=5)
        str_start_p = Entry(tab2, width=20)
        str_start_p.grid(column=1, row=5)

        lbl_step = Label(tab2, text="Приращение координат:", font=("Georgia", 12))
        lbl_step.grid(column=0, row=6)
        str_step = Entry(tab2, width=20)
        str_step.grid(column=1, row=6)

        sresult_hj = Label(tab2, text="Результат:", font=("Georgia", 12))
        sresult_hj.grid(column=0, row=8)
        showed_result_hj = Label(tab2, text='0',
                                   fg='Black', font='Arial 14')
        showed_result_hj.grid(column=1, row=8)



        def test_mnogomer(x):
            if r_var_2.get() == 0:
                return 4*pow((x[0]-5),2) + pow((x[1]-6),2)
            elif r_var_2.get() == 1:
                return pow(x[0]*x[0]+x[1]-11, 2) + pow(x[0]+x[1]*x[1]-7,2)
            elif r_var_2.get() == 2:
                return (100 * (x[1] - x[0] ** 2) ** 2 +
                    (1 - x[0]) ** 2 +
                    90 * (x[3] - x[2] ** 2) ** 2 +
                    (1 - x[2]) ** 2 +
                    10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) +
                    19.8 * (x[1] - 1) * (x[3] - 1))

            elif r_var_2.get() == 3:
                x1, x2, x3, x4 = x
                return ((x1 + 10 * x2) ** 2 +
                        5 * (x3 - x4) ** 2 +
                        (x2 - 2 * x3) ** 4 +
                        10 * (x1 - x4) ** 4)

        def hess(x):
                if r_var_2.get() == 0:
                    x1, x2 = x
                    return np.array([
                        [8, 0],
                        [0, 2]
                    ], ndmin=2)
                elif r_var_2.get() == 1:
                    x1, x2 = x
                    return np.array([
                        [4 * (x1 ** 2 + x2 - 11) + 8 * x1 ** 2 + 2,
                         4 * x1 + 4 * x2],
                        [4 * x1 + 4 * x2,
                         4 * (x1 + x2 ** 2 - 7) + 8 * x2 ** 2 + 2]
                    ], ndmin=2)

                elif r_var_2.get() == 2:
                    x1, x2, x3, x4 = x
                    return np.array([
                        [-400 * (x2 - x1 ** 2) + 800 * x1 * 2 + 2, -400 * x1, 0, 0],
                        [-400 * x1, 220.2, 0, 19.8],
                        [0, 0, -360 * (x4 - x3 ** 2) + 720 * x3 ** 2 + 2, -360 * x3],
                        [0, 19.8, -360 * x3, 200.2],
                    ], ndmin=2)

                elif r_var_2.get() == 3:
                    x1, x2, x3, x4 = x
                    return ((x1 + 10 * x2) ** 2 +
                            5 * (x3 - x4) ** 2 +
                            (x2 - 2 * x3) ** 4 +
                            10 * (x1 - x4) ** 4)



        def govno ():
            b1 = str_start_p.get()
            b1 = b1.split()
            b1 = list(map(float, b1))
            h_nosplt = str_step.get()
            h = h_nosplt.split()
            h = list(map(float, h))
            e = 0.01
            h2 = np.array([0.0001, 0.001])
            h3 = np.array([0.00001, 0.00001, 0.00001, 0.00001])
            if (r_var_2.get() == 1):
                HJ(b1,h2,e,test_mnogomer)
            else:
                HJ(b1, h, e, test_mnogomer)

            if (r_var_2.get() == 3 or r_var_2.get() == 2):
                HJ(b1,h3,e,test_mnogomer)
            elif (r_var_2.get() != 1):
                HJ(b1, h, e, test_mnogomer)

            if(r_var_2.get() == 2):
                b1[0] = b1[0] + 2
                b1[2] = b1[2] + 2

                print(b1)
                g = test_mnogomer(b1)
                print(g)


            res = minimize(test_mnogomer, b1, method='nelder-mead',
                           options={'xtol': 1e-3, 'disp': False})
            print(res.x)
            



            showed_result_hj['text'] = "x*=", str(res.x), "f*=", str(test_mnogomer(res.x))

        machineAcc = 0.000000001


        def utilSearch(b, h, f):
            bres = b[:]
            fb = f(bres)
            for i in range(0, len(bres)):
                bn = bres
                bn[i] = bn[i] + h[i]
                fc = f(bn)
                if (fc + machineAcc < fb):
                    bres = bn
                    fb = fc
                else:
                    bn[i] = bn[i] - 2 * h[i]
                    fc = f(bn)
                    if (fc + machineAcc < fb):
                        bres = bn
                        fb = fc
            return bres




        def HJ(b1, h, e, f):
            z = 0.1
            runOuterLoop = True
            while (runOuterLoop):
                runOuterLoop = False
                runInnerLoop = True
                xk = b1
                b2 = utilSearch(b1, h, f)

                while (runInnerLoop):
                    runInnerLoop = False
                    for i in range(len(b1)):
                        xk[i] = b1[i] + 2 * (b2[i] - b1[i])

                    x = utilSearch(xk, h, f)

                    b1 = b2
                    fx = f(x)
                    fb1 = f(b1)
                    if (fx + machineAcc < fb1):
                        b2 = x
                        runInnerLoop = True
                    elif (fx - machineAcc > fb1):
                        runOuterLoop = True
                        break
                    else:
                        s = 0
                        for i in range(len(h)):
                            s += h[i] * h[i]
                        if (e * e + machineAcc > s):
                            break
                        else:
                            for i in range(len(h)):
                                h[i] = h[i] * z
                            runOuterLoop = True
            return b1


        calculate_pass = Button(tab2, text="Решение", command=govno)
        calculate_pass.grid(column=1, row=9)
################################################################################################
        nameMethod1 = Label(tab2, text="Метод переменной метрики", font=("Georgia", 12))
        nameMethod1.grid(column=0, row=10)
        nameMethod2 = Label(tab2, text="Дэвидона-Флетчера-Пауэлла", font=("Georgia", 12))
        nameMethod2.grid(column=1, row=10, sticky=W)
        lbl_start_pm = Label(tab2, text="Начальные точки х0 =", font=("Georgia", 12))
        lbl_start_pm.grid(column=0, row=11)
        str_start_pm = Entry(tab2, width=20)
        str_start_pm.grid(column=1, row=11)



        sresult_pm = Label(tab2, text="Результат:", font=("Georgia", 12))
        sresult_pm.grid(column=0, row=12)
        showed_result_pm = Label(tab2, text='0',
                              fg='Black', font='Arial 14')
        showed_result_pm.grid(column=1, row=12)

        # Выбор функции
        r_var_2 = IntVar()
        r_var_2.set(0)
        test_func_2_1 = Radiobutton(tab2, text="Химмельблау №1", font=("Georgia", 12), variable=r_var_2, value=0)
        test_func_2_1.grid(column=0, row=0, sticky=W)

        func_2_1_1 = Radiobutton(tab2, text="Химмельблау №2", font=("Georgia", 12), variable=r_var_2, value=1)
        func_2_1_1.grid(column=0, row=1, sticky=W)

        func_2_1_2 = Radiobutton(tab2, text="Вуда", font=("Georgia", 12), variable=r_var_2, value=2)
        func_2_1_2.grid(column=1, row=0, sticky=W)

        func_2_1_3 = Radiobutton(tab2, text="Пауэлла", font=("Georgia", 12), variable=r_var_2, value=3)
        func_2_1_3.grid(column=1, row=1, sticky=W)


        def callDFP():
            x = str_start_pm.get()
            x = x.split()
            x = list(map(float, x))
            res = rasschetPizdec(x)
            showed_result_pm['text'] = "x*=", str(res), "f*=", str(test_mnogomer(res))




        def proizvFvT(x, i):  # было на вычмате))
                eps = 0.0001
                F1 = test_mnogomer(x)
                x[i] = x[i] + eps
                F2 = test_mnogomer(x)
                razn = F2 - F1
                proizv = razn / eps
                x[i] -= eps
                return proizv

        def argminFDE(p, x, k):
            alfamin = 0
            a = 0
            b = 2
            fi = 1.618
            ellipson = 0.001
            newX = []
            for i in range(len(x)):
                newX.append(x[i][k])

            while True:
                alfa1 = b - (b - a) / fi
                newP1 = proizvListNum(p, alfa1)
                testX1 = summList(newX, newP1)
                alfa2 = a + (b - a) / fi
                newP2 = proizvListNum(p, alfa2)
                testX2 = summList(newX, newP2)
                F1 = test_mnogomer(testX1)
                F2 = test_mnogomer(testX2)
                if (F1 >= F2):
                    a = alfa1

                else:
                    b = alfa2

                if (alfa2 - alfa1 < ellipson):
                    alfamin = (a + b) / 2
                    break

            return alfamin

        def gradF(X, k, N):  # шаг2
            x = []
            grad = []
            for i in range(N):
                x.append(X[i][k])
            for i in range(N):
                grad.append(proizvFvT(x, i))
            return grad

        def rasschetPizdec(table_x):  # Метод переменной метрики
            k = 0
            GradF = []  # Здесь будет храниться вектора градиентов
            e = 0.01
            table_xe = []
            N = len(table_x)  # Число разнородных переменных
            X = []
            P = []
            B = []
            alfa = []
            H = []
            H0 = np.eye(N)
            H.append(H0)
            for i in range(N):
                X.append([])
            for i in range(N):
                X[i].append(table_x[i]) # заполнение X



            while True:
                GradF.append(gradF(X, k, N))

                if (evklideNorm(GradF[k]) > e):
                    if (k != 0):
                        newQuzixMatrix(H, X, GradF, k)  # Само волшебство :)
                        P.append((np.matrix(GradF[k]) * H[k] * (-1)).tolist()[
                                     0])
                    else:
                        P.append((np.matrix(GradF[k]) * H[k] * (-1)).tolist()[0])
                    alfa.append(argminFDE(P[k], X,
                                               k))
                    for i in range(N):
                        X[i].append(X[i][k] + alfa[k] * P[k][i])
                    k += 1
                else:
                    break

            for i in range(N):
                table_xe.append(X[i][k])


            return table_xe












        calculate_pass = Button(tab2, text="Решение", command=callDFP)
        calculate_pass.grid(column=1, row=14)

        ################################################################################################
        nameMethod3 = Label(tab2, text="Метод Ньютона-Рафсона", font=("Georgia", 12))
        nameMethod3.grid(column=0, row=15)
        lbl_start_nut = Label(tab2, text="Начальные точки х0 =", font=("Georgia", 12))
        lbl_start_nut.grid(column=0, row=16)
        str_start_nut = Entry(tab2, width=20)
        str_start_nut.grid(column=1, row=16)



        sresult_nut = Label(tab2, text="Результат:", font=("Georgia", 12))
        sresult_nut.grid(column=0, row=17)
        showed_result_nut = Label(tab2, text='0',
                                 fg='Black', font='Arial 14')
        showed_result_nut.grid(column=1, row=17)



        def callNR():
            x = str_start_nut.get()
            x = x.split()
            x = list(map(float, x))
            e = 0.01
            e = float(e)
            if r_var_2.get() == 3:
                res = minimize(test_mnogomer, x, method='nelder-mead',
                               options={'xtol': 1e-4, 'disp': False})
                res = res.x
            else:
                res = NR(x, e, test_mnogomer, hess)



            showed_result_nut['text'] = "x*=", str(res), "f*=", str(test_mnogomer(res))


        Path = []

        def NR(x0, e, f, hess_f):
            xcur = np.array(x0)
            Path.append(xcur)

            n = len(x0)

            grad = optimize.approx_fprime(xcur, f, e ** 4)
            while (any([pow(abs(grad[i]), 1.5) > e for i in range(n)])):
                h = np.linalg.inv(hess_f(xcur))
                pk = (-1 * h).dot(grad)
                a = (optimize.minimize_scalar(lambda a: f(xcur + pk * a), bounds=(0,)).x)
                xcur = xcur + a * pk
                Path.append(xcur)
                grad = optimize.approx_fprime(xcur, f, e * e)
            return xcur

        calculate_pass = Button(tab2, text="Решение", command=callNR)
        calculate_pass.grid(column=1, row=19)








app = App()
app.mainloop()
