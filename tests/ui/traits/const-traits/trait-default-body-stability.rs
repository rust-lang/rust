//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(staged_api)]
#![feature(const_trait_impl)]
#![feature(const_t_try)]
#![feature(const_try)]
#![feature(try_trait_v2)]
#![stable(feature = "foo", since = "1.0")]

use std::ops::{ControlFlow, TryFromBreak, Try, TryAs};

#[stable(feature = "foo", since = "1.0")]
pub struct T;

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_t_try", issue = "none")]
const impl Try for T {
    type Output = T;
    type Break = T;

    fn from_output(t: T) -> T {
        t
    }

    fn branch(self) -> ControlFlow<T, T> {
        ControlFlow::Continue(self)
    }
}

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_t_try", issue = "none")]
const impl TryAs<T> for T {
    type TryType = T;
}

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_t_try", issue = "none")]
const impl TryFromBreak for T {
    fn from_break(t: T) -> T {
        t
    }
}

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_tr", issue = "none")]
pub const trait Tr {
    #[stable(feature = "foo", since = "1.0")]
    fn bar() -> T {
        T?
        // Should be allowed.
        // Must enable unstable features to call this trait fn in const contexts.
    }
}

fn main() {}
