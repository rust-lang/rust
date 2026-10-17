#![feature(rustc_attrs)]

#![deny(dead_code)]

trait Trait {
    fn bar(&self) -> usize;
}

struct Foo(usize); //~ ERROR: field `0` is never read

impl Trait for Foo {
    #[rustc_trivial_field_reads]
    fn bar(&self) -> usize {
        self.0
    }
}

fn main() {
    println!("{}", Foo(0).bar());
}
