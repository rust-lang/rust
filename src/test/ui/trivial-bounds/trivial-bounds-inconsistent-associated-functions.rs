// run-pass
// Inconsistent bounds with trait implementations

#![feature(trivial_bounds)]
#![allow(unused)]

trait A {
    fn foo(&self) -> Self where Self: Copy;
}

impl A for str {
    fn foo(&self) -> Self where Self: Copy { *"" }
}

impl A for i32 {
    fn foo(&self) -> Self { 3 }
}

fn main() {}
