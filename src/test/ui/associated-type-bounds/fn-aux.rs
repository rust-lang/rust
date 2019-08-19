// run-pass
// aux-build:fn-aux.rs

#![feature(associated_type_bounds)]

extern crate fn_aux;

use fn_aux::*;

fn main() {
    desugared();
}
