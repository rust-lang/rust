//! Regression test for <https://github.com/rust-lang/rust/issues/135772>.
// Unknown diagnostic attributes must respect item-local lint levels.

//@ check-pass

trait _Trait {}

#[allow(unknown_or_malformed_diagnostic_attributes)]
#[diagnostic::abcdef]
impl _Trait for () {}

#[diagnostic::abcdef]
#[allow(unknown_or_malformed_diagnostic_attributes)]
impl _Trait for bool {}

#[expect(unknown_or_malformed_diagnostic_attributes)]
#[diagnostic::abcdef]
impl _Trait for u8 {}

macro_rules! impl_trait {
    ($ty:ty) => {
        #[allow(unknown_or_malformed_diagnostic_attributes)]
        #[diagnostic::abcdef]
        impl _Trait for $ty {}
    };
}

impl_trait!(u16);

#[diagnostic::abcdef]
//~^ WARN unknown diagnostic attribute
impl _Trait for u32 {}

fn main() {}
