//@ run-pass
//! Test that guard patterns in union fields don't impose an `unsafe` requirement.

#![feature(guard_patterns)]
#![expect(incomplete_features)]

union Foo {
    field: u8,
}

fn main() {
    let foo = Foo { field: 42 };
    match foo {
        Foo { field: _ if false } => (),
        _ => panic!(), //~ WARN unreachable
    }
}
