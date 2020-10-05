#![feature(const_fn, type_alias_impl_trait)]

type Foo = impl Send;

// This is not structural-match
struct A;

const fn value() -> Foo {
    A
}
const VALUE: Foo = value();

fn test() {
    match todo!() {
        VALUE => (),
        //~^ `impl Send` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
