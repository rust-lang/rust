//@ check-pass
//@ compile-flags: --emit=mir,link -Zmir-opt-level=4
// Checks that we don't ICE due to attempting to run const prop
// on a function with unsatisifable 'where' clauses

#![allow(unused)]

trait A {
    fn foo(&self) -> Self where Self: Copy;
}

impl A for [fn(&())] {
    fn foo(&self) -> Self where Self: Copy { *(&[] as &[_]) }
}

impl A for i32 {
    fn foo(&self) -> Self { 3 }
}

fn main() {}
