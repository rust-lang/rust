// This test checks whether an item which is used trivially
// from a normal seed is then checked again when coming from
// a allow/expect seed, which should mark it as non-trivially
// used.
//
//@ check-pass

#![feature(rustc_attrs)]
#![deny(dead_code)]

trait Trait {
    fn bar(&self);
}

struct Foo(usize);

fn print() {
    println!("{}", Foo(0).0);
}

impl Trait for Foo {
    #[rustc_trivial_field_reads]
    fn bar(&self) {
        print();
    }
}

#[expect(dead_code)]
fn entry() {
    print();
}

fn main() {
    Foo(0).bar();
}
