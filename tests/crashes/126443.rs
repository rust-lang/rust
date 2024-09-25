//@ known-bug: #126443
//@ compile-flags: -Copt-level=0
#![feature(generic_const_exprs)]

fn double_up<const M: usize>() -> [(); M * 2] {
    todo!()
}

fn quadruple_up<const N: usize>() -> [(); N * 2 * 2] {
    double_up()
}

fn main() {
    quadruple_up::<0>();
}
