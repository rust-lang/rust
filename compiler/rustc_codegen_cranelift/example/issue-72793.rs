// Adapted from rustc ui test suite (ui/type-alias-impl-trait/issue-72793.rs)

#![feature(type_alias_impl_trait)]

mod helper {
    pub trait T {
        type Item;
    }

    pub type Alias<'a> = impl T<Item = &'a ()>;

    struct S;
    impl<'a> T for &'a S {
        type Item = &'a ();
    }

    pub fn filter_positive<'a>() -> Alias<'a> {
        &S
    }
}
use helper::*;

fn with_positive(fun: impl Fn(Alias<'_>)) {
    fun(filter_positive());
}

fn main() {
    with_positive(|_| ());
}
