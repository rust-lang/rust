#![feature(rustc_attrs)]

trait Foo { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
impl<T: 'static> Foo for T where T: Iterator<Item = i32> { }

trait Bar {
    type Assoc;
}

impl<T> Bar for T where T: Iterator<Item = i32> {
    #[rustc_dump_program_clauses] //~ ERROR program clause dump
    type Assoc = Vec<T>;
}

fn main() {
    println!("hello");
}
