#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Foo {
    #[rustc_dump_env_program_clauses] //~ ERROR program clause dump
    fn foo(&self);
}

impl<T> Foo for T where T: Clone {
    #[rustc_dump_env_program_clauses] //~ ERROR program clause dump
    fn foo(&self) {
    }
}

fn main() {
}
