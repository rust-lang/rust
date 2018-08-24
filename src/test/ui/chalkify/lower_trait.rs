#![feature(rustc_attrs)]

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Foo<S, T, U> {
    fn s(S) -> S;
    fn t(T) -> T;
    fn u(U) -> U;
}

fn main() {
    println!("hello");
}
