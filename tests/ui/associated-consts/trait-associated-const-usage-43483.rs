// https://github.com/rust-lang/rust/issues/43483
//@ check-pass
#![allow(dead_code)]
#![allow(unused_variables)]
trait VecN {
    const DIM: usize;
}

trait Mat {
    type Row: VecN;
}

fn m<M: Mat>() {
    let x = M::Row::DIM;
}

fn main() {}
