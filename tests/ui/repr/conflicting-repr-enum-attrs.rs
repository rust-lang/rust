//! Regression test for <https://github.com/rust-lang/rust/issues/47094>.
//! Test `conflicting representation hints` warning is being triggered
//! when there are multiple repr attributes.

#[repr(C, u8)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
enum Foo {
    A,
    B,
}

#[repr(C)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
#[repr(u8)]
enum Bar {
    A,
    B,
}

fn main() {}
