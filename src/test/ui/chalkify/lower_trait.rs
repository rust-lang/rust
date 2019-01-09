#![feature(rustc_attrs)]

trait Bar { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Foo<S, T: ?Sized> {
    #[rustc_dump_program_clauses] //~ ERROR program clause dump
    type Assoc: Bar + ?Sized;
}

fn main() {
    println!("hello");
}
