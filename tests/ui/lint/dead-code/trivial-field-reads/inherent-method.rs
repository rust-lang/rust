#![feature(rustc_attrs)]

#![deny(dead_code)]

struct Foo(usize); //~ ERROR: field `0` is never read

impl Foo {
    #[rustc_trivial_field_reads]
    fn bar(&self) -> usize {
        self.0
    }
}

fn main() {
    println!("{}", Foo(0).bar());
}
