#![feature(type_alias_impl_trait)]

// check-pass

type Foo = impl std::fmt::Debug;

struct Wrap(Foo);

// Check that fields count as defining uses, too
fn h() -> Wrap {
    Wrap(42)
}

fn main() {}
