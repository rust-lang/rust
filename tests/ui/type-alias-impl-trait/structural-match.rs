#![feature(type_alias_impl_trait)]

pub type Foo = impl Send;

// This is not structural-match
struct A;

#[defines(Foo)]
pub const fn value() -> Foo {
    A
}

const VALUE: Foo = value();

fn test() {
    match VALUE {
        VALUE => (),
        //~^ `Foo` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
