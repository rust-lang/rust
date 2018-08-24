#![feature(rustc_attrs)]

use std::fmt::{Debug, Display};
use std::borrow::Borrow;

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Foo<'a, 'b, S, T, U> where S: Debug, T: Borrow<U>, U: ?Sized, 'a: 'b, U: 'b {
    fn s(S) -> S;
    fn t(T) -> T;
    fn u(U) -> U;
}

fn main() {
    println!("hello");
}
