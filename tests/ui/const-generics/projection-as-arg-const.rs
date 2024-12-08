// This is currently not possible to use projections as const generics.
// More information about this available here:
// https://github.com/rust-lang/rust/pull/104443#discussion_r1029375633

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub fn foo<const X: <i32 as Identity>::Identity>() {
//~^ ERROR
    assert!(X == 12);
}

fn main() {
    foo::<12>();
}
