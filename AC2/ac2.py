import matplotlib.pyplot as plt
import numpy as np

def centralizar(pontos):
    x_min, y_min = np.min(pontos, axis=0)
    x_max, y_max = np.max(pontos, axis=0)
    
    largura = x_max - x_min
    altura = y_max - y_min
    
    margem = 0.1
    margem_x = largura * margem
    margem_y = altura * margem
    
    plt.xlim(x_min - margem_x, x_max + margem_x)
    plt.ylim(y_min - margem_y, y_max + margem_y)
    
    plt.gca().set_aspect('equal', adjustable='box')

def questao1():
    # Ponto original
    P = (2, 3)
    # Vetor de translação
    T = (4, -2)
    # Novo ponto após translação
    P_linha = (P[0] + T[0], P[1] + T[1])

    # Plotando
    plt.figure()
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Pontos
    plt.plot(P[0], P[1], 'bo', label='Ponto antes')
    plt.plot(P_linha[0], P_linha[1], 'ro', label="Ponto depois")

    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.legend()
    plt.title("Questão 1: Translação Simples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def questao2():
    # Vértices originais
    A = (1, 1)
    B = (3, 1)
    C = (2, 4)

    # Fator de escala
    s = 2

    # Vértices após escala
    A_ = (A[0]*s, A[1]*s)
    B_ = (B[0]*s, B[1]*s)
    C_ = (C[0]*s, C[1]*s)

    # Plotando
    plt.figure()
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Triângulo original
    x_orig = [A[0], B[0], C[0], A[0]]
    y_orig = [A[1], B[1], C[1], A[1]]
    plt.plot(x_orig, y_orig, 'b-', label='Triângulo Original')

    # Triângulo escalado
    x_scaled = [A_[0], B_[0], C_[0], A_[0]]
    y_scaled = [A_[1], B_[1], C_[1], A_[1]]
    plt.plot(x_scaled, y_scaled, 'r--', label='Triângulo Escalado (s=2)')

    # Marcando pontos
    plt.plot(*A, 'bo')
    plt.text(A[0], A[1]-0.3, 'A', ha='center')
    plt.plot(*B, 'bo')
    plt.text(B[0], B[1]-0.3, 'B', ha='center')
    plt.plot(*C, 'bo')
    plt.text(C[0], C[1]+0.3, 'C', ha='center')

    plt.plot(*A_, 'ro')
    plt.text(A_[0], A_[1]-0.3, "A'", ha='center')
    plt.plot(*B_, 'ro')
    plt.text(B_[0], B_[1]-0.3, "B'", ha='center')
    plt.plot(*C_, 'ro')
    plt.text(C_[0], C_[1]+0.3, "C'", ha='center')

    plt.title("Questão 2: Escala Uniforme (Fator 2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.show()

def questao3():
    # Vértices originais
    A = (1, 1)
    B = (3, 1)
    C = (2, 4)

    # Fatores de escala
    sx = 2
    sy = 0.5

    # escala não uniforme
    A_ = (A[0]*sx, A[1]*sy)
    B_ = (B[0]*sx, B[1]*sy)
    C_ = (C[0]*sx, C[1]*sy)

    # Plotando
    plt.figure()
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Triângulo original
    x_orig = [A[0], B[0], C[0], A[0]]
    y_orig = [A[1], B[1], C[1], A[1]]
    plt.plot(x_orig, y_orig, 'b-', label='Triângulo Original')

    # Triângulo escalado
    x_scaled = [A_[0], B_[0], C_[0], A_[0]]
    y_scaled = [A_[1], B_[1], C_[1], A_[1]]
    plt.plot(x_scaled, y_scaled, 'r--', label='Escala Não Uniforme (sx=2, sy=0.5)')

    # Pontos
    for p, label in zip([A, B, C], ['A', 'B', 'C']):
        plt.plot(*p, 'bo')
        plt.text(p[0], p[1]-0.3, label, ha='center')

    for p, label in zip([A_, B_, C_], ["A'", "B'", "C'"]):
        plt.plot(*p, 'ro')
        plt.text(p[0], p[1]-0.3, label, ha='center')

    plt.title("Questão 3: Escala Não Uniforme (sx=2, sy=0.5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.show()

def questao4():
    # Ponto original
    x, y = 1, 0
    # Ângulo de rotação em radianos
    theta = np.pi / 2  # 90 graus

    # Aplicando rotação
    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)

    # Visualização
    plt.figure(figsize=(6,6))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    # Pontos
    plt.plot(x, y, 'bo', label='P(1, 0)')
    plt.plot(x_prime, y_prime, 'ro', label="P'(0, 1)")

    # Vetor do ponto original
    plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    # Vetor do ponto rotacionado
    plt.arrow(0, 0, x_prime, y_prime, head_width=0.05, head_length=0.1, fc='red', ec='red')

    # Rótulos
    plt.text(x+0.1, y-0.1, 'P(1,0)', fontsize=12, color='blue')
    plt.text(x_prime+0.1, y_prime, "P'(0,1)", fontsize=12, color='red')

    plt.grid(True)
    plt.legend()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title("Questão 4: Rotação de P(1,0) em 90° anti-horário")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def questao5():
    # Vértices originais
    vertices = np.array([
        [1, 1],
        [1, 4],
        [4, 4],
        [4, 1]
    ])

    # Ângulo de rotação em radianos (45° horário = -45° anti-horário)
    theta = -np.pi / 4

    # Matriz de rotação
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Aplica rotação a cada vértice
    rotacionados = vertices @ R.T  # multiplicação de matriz

    # Visualização
    plt.figure(figsize=(8,8))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True)

    # Polígono original (fechado)
    x_orig = np.append(vertices[:,0], vertices[0,0])
    y_orig = np.append(vertices[:,1], vertices[0,1])
    plt.plot(x_orig, y_orig, 'b-', label='Original')

    # Polígono rotacionado (fechado)
    x_rot = np.append(rotacionados[:,0], rotacionados[0,0])
    y_rot = np.append(rotacionados[:,1], rotacionados[0,1])
    plt.plot(x_rot, y_rot, 'r-', label='Rotacionado')

    # Pontos originais e rotacionados com rótulos
    for i, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'bo')
        plt.text(x + 0.1, y, f'{chr(65+i)}({x:.2f},{y:.2f})', color='blue')

    for i, (x, y) in enumerate(rotacionados):
        plt.plot(x, y, 'ro')
        plt.text(x + 0.1, y, f"{chr(65+i)}'({x:.2f},{y:.2f})", color='red')

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Questão 5: Rotação de 45° horário de um quadrado')
    plt.show()

    # Retorna as coordenadas rotacionadas para a resposta
    return rotacionados

def questao6():
    # Ponto original
    x, y = 2, 5
    # Ponto refletido em relação ao eixo y
    x_prime, y_prime = -x, y

    # Visualização
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    # Pontos
    plt.plot(x, y, 'bo', label='P(2, 5)')
    plt.plot(x_prime, y_prime, 'ro', label="P'(-2, 5)")

    # Linhas auxiliadoras
    plt.plot([x, x_prime], [y, y_prime], 'g--', label='Reflexão')

    # Rótulos
    plt.text(x + 0.2, y, 'P(2,5)', fontsize=12, color='blue')
    plt.text(x_prime + 0.2, y_prime, "P'(-2,5)", fontsize=12, color='red')

    plt.grid(True)
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(0, 6)
    plt.title("Questão 6: Reflexão de P(2,5) em relação ao eixo y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def questao7():
    # Vértices originais do triângulo
    vertices = np.array([
        [2, 3],
        [4, 3],
        [3, 5]
    ])

    # Reflexão em relação ao eixo x
    reflexao = vertices.copy()
    reflexao[:, 1] = -reflexao[:, 1]

    # Visualização
    plt.figure(figsize=(8, 6))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True)

    # Triângulo original (fechado)
    x_orig = np.append(vertices[:,0], vertices[0,0])
    y_orig = np.append(vertices[:,1], vertices[0,1])
    plt.plot(x_orig, y_orig, 'b-', label='Original')

    # Triângulo refletido (fechado)
    x_ref = np.append(reflexao[:,0], reflexao[0,0])
    y_ref = np.append(reflexao[:,1], reflexao[0,1])
    plt.plot(x_ref, y_ref, 'r-', label='Refletido')

    # Pontos com rótulos
    for i, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'bo')
        plt.text(x + 0.1, y, f'{chr(65+i)}({x},{y})', color='blue')

    for i, (x, y) in enumerate(reflexao):
        plt.plot(x, y, 'ro')
        plt.text(x + 0.1, y, f"{chr(65+i)}'({x},{y})", color='red')

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Questão 7: Reflexão em relação ao eixo x de um triângulo')
    plt.show()

    # Retorna as coordenadas refletidas para resposta
    return reflexao

def questao8():
    # Ponto original
    x, y = 2, 3
    k = 2  # fator de cisalhamento horizontal

    # Aplica cisalhamento horizontal
    x_prime = x + k * y
    y_prime = y

    # Visualização
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    # Pontos
    plt.plot(x, y, 'bo', label='P(2, 3)')
    plt.plot(x_prime, y_prime, 'ro', label="P'(8, 3)")

    # Linha mostrando o movimento
    plt.plot([x, x_prime], [y, y_prime], 'g--', label='Cisalhamento')

    # Rótulos
    plt.text(x + 0.2, y, 'P(2,3)', fontsize=12, color='blue')
    plt.text(x_prime + 0.2, y_prime, "P'(8,3)", fontsize=12, color='red')

    plt.grid(True)
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.title("Questão 8: Cisalhamento horizontal de P(2,3) com k=2")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def questao9():
    # Ponto inicial
    x, y = 3, 2

    # 1. Translação
    dx, dy = 1, -1
    x_t, y_t = x + dx, y + dy

    # 2. Rotação 90° anti-horário
    theta = np.pi / 2
    x_r = x_t * np.cos(theta) - y_t * np.sin(theta)
    y_r = x_t * np.sin(theta) + y_t * np.cos(theta)

    # 3. Escala uniforme
    s = 2
    x_final, y_final = s * x_r, s * y_r

    # Visualização
    plt.figure(figsize=(7,7))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True)

    # Ponto inicial
    plt.plot(x, y, 'bo', label='P(3,2)')
    plt.text(x + 0.1, y, 'P(3,2)', color='blue')

    # Após translação
    plt.plot(x_t, y_t, 'co', label='Após Translação')
    plt.text(x_t + 0.1, y_t, f'({x_t:.1f},{y_t:.1f})', color='cyan')

    # Após rotação
    plt.plot(x_r, y_r, 'mo', label='Após Rotação')
    plt.text(x_r + 0.1, y_r, f'({x_r:.1f},{y_r:.1f})', color='magenta')

    # Após escala
    plt.plot(x_final, y_final, 'ro', label="P' Final")
    plt.text(x_final + 0.1, y_final, f"({x_final:.1f},{y_final:.1f})", color='red')

    plt.legend()
    plt.title("Questão 9: Composição de transformações")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-5, 10)
    plt.ylim(-1, 10)
    plt.show()

    return (x_final, y_final)

def questao10():
    # Vértices originais
    vertices = np.array([
        [1, 1],
        [5, 1],
        [5, 3],
        [1, 3]
    ])

    # 1. Translação
    trans_vector = np.array([-2, 3])
    vertices_t = vertices + trans_vector

    # 2. Escala não uniforme
    scale_factors = np.array([1.5, 0.5])
    vertices_s = vertices_t * scale_factors

    # 3. Reflexão em relação ao eixo y
    vertices_r = vertices_s.copy()
    vertices_r[:, 0] = -vertices_r[:, 0]

    # Visualização
    plt.figure(figsize=(8,8))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True)

    # Polígono original (fechado)
    x_orig = np.append(vertices[:,0], vertices[0,0])
    y_orig = np.append(vertices[:,1], vertices[0,1])
    plt.plot(x_orig, y_orig, 'b-', label='Original')

    # Polígono transformado (fechado)
    x_tr = np.append(vertices_r[:,0], vertices_r[0,0])
    y_tr = np.append(vertices_r[:,1], vertices_r[0,1])
    plt.plot(x_tr, y_tr, 'r-', label='Transformado')

    # Pontos originais com rótulos
    for i, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'bo')
        plt.text(x + 0.1, y, f'{chr(65+i)}({x},{y})', color='blue')

    # Pontos transformados com rótulos
    for i, (x, y) in enumerate(vertices_r):
        plt.plot(x, y, 'ro')
        plt.text(x + 0.1, y, f"{chr(65+i)}'({x:.2f},{y:.2f})", color='red')

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Questão 10: Transformações em sequência em um retângulo')
    plt.show()

    # Retorna as coordenadas transformadas para resposta
    return vertices_r

questao2()
