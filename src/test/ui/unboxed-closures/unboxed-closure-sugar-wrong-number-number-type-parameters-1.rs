#![feature(unboxed_closures)]

trait One<A> { fn foo(&self) -> A; }

fn foo(_: &One()) //~ ERROR associated type `Output` not found for `One<()>`
{}

fn main() { }
