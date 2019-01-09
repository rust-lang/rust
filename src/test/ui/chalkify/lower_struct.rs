#![feature(rustc_attrs)]

#[rustc_dump_program_clauses] //~ ERROR program clause dump
struct Foo<'a, T> where Box<T>: Clone {
    _x: std::marker::PhantomData<&'a T>,
}

fn main() { }
