#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Foo { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Bar where Self: Foo { }

#[rustc_dump_env_program_clauses] //~ ERROR program clause dump
fn bar<T: Bar + ?Sized>() {
}

fn main() {
}
