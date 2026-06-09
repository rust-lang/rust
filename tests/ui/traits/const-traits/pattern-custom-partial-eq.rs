//! Ensure that a `const fn` can match on constants of a type that is `PartialEq`
//! but not `const PartialEq`. This is accepted for backwards compatibility reasons.
//@ check-pass
#![feature(const_trait_impl)]

#[derive(Eq, PartialEq)]
pub struct Y(u8);
pub const GREEN: Y = Y(4);
pub const fn is_green(x: Y) -> bool {
    match x { GREEN => true, _ => false }
}

struct CustomEq;

impl Eq for CustomEq {}
impl PartialEq for CustomEq {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

#[derive(PartialEq, Eq)]
#[allow(unused)]
enum Foo {
    Bar,
    Baz,
    Qux(CustomEq),
}

const BAR_BAZ: Foo = if 42 == 42 {
    Foo::Bar
} else {
    Foo::Qux(CustomEq) // dead arm
};

const EMPTY: &[CustomEq] = &[];

const fn test() {
    // BAR_BAZ itself is fine but the enum has other variants
    // that are non-structural. Still, this should be accepted.
    match Foo::Qux(CustomEq) {
        BAR_BAZ => panic!(),
        _ => {}
    }

    // Similarly, an empty slice of a type that is non-structural
    // is accepted.
    match &[CustomEq] as &[CustomEq] {
        EMPTY => panic!(),
        _ => {},
    }
}

fn main() {}
