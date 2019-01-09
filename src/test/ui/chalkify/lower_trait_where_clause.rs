#![feature(rustc_attrs)]

use std::borrow::Borrow;

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Foo<'a, 'b, T, U>
where
    T: Borrow<U> + ?Sized,
    U: ?Sized + 'b,
    'a: 'b,
    Box<T>:, // NOTE(#53696) this checks an empty list of bounds.
{
}

fn main() {
    println!("hello");
}
