#![feature(rustc_attrs)]

#[rustc_dump_program_clauses] //~ ERROR program clause dump
struct Foo<T> where Box<T>: Clone {
    _x: std::marker::PhantomData<T>,
}

fn main() { }
