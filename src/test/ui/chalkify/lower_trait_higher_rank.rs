#![feature(rustc_attrs)]

#[rustc_dump_program_clauses] //~ ERROR program clause dump
trait Foo<F: ?Sized> where for<'a> F: Fn(&'a (u8, u16)) -> &'a u8
{
}

fn main() {
    println!("hello");
}
