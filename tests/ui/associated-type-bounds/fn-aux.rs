//@ run-pass
//@ aux-build:fn-aux.rs

extern crate fn_aux;

use fn_aux::*;

fn main() {
    desugared();
}
