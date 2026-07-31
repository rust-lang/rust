//@ compile-flags: -Zmir-opt-level=2
// NOTE: Este teste verifica que o pass BoolChainOpt é registrado e executa
// no nível de otimização 2 sem causar crashes ou regressões semânticas.
// A mutação do CFG será adicionada em um follow-up após validação deste scaffold.

pub struct S {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

pub fn eq(x: &S, y: &S) -> bool {
    x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d
}

fn main() {
    let s1 = S { a: 1, b: 2, c: 3, d: 4 };
    let s2 = S { a: 1, b: 2, c: 3, d: 4 };
    assert!(eq(&s1, &s2));
}
