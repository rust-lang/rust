//@ known-bug: #117460
#![feature(generic_const_exprs)]

struct Matrix<D = [(); 2 + 2]> {
    d: D,
}

impl Matrix {}
