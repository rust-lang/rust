#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Foo { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
struct S<'a, T: ?Sized> where T: Foo {
    data: &'a T,
}

#[rustc_dump_env_program_clauses] //~ ERROR program clause dump
fn bar<T: Foo>(_x: S<'_, T>) { // note that we have an implicit `T: Sized` bound
}

fn main() {
}
