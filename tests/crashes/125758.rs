//@ known-bug: rust-lang/rust#125758
#![feature(impl_trait_in_assoc_type)]

trait Trait: Sized {
    type Assoc2;
}

impl Trait for Bar {
    type Assoc2 = impl std::fmt::Debug;
}

struct Foo {
    field: <Bar as Trait>::Assoc2,
}

enum Bar {
    C = 42,
    D = 99,
}

static BAR: u8 = 42;

static FOO2: (&Foo, &<Bar as Trait>::Assoc2) =
    unsafe { (std::mem::transmute(&BAR), std::mem::transmute(&BAR)) };

fn main() {}
