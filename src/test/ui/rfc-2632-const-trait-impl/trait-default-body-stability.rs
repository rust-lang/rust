// check-pass

#![feature(staged_api)]
#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]
#![feature(const_t_try)]
#![feature(const_try)]
#![feature(try_trait_v2)]

#![stable(feature = "foo", since = "1.0")]

use std::ops::{ControlFlow, FromResidual, Try};

#[stable(feature = "foo", since = "1.0")]
pub struct T;

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_t_try", issue = "none")]
impl const Try for T {
    type Output = T;
    type Residual = T;

    fn from_output(t: T) -> T {
        t
    }

    fn branch(self) -> ControlFlow<T, T> {
        ControlFlow::Continue(self)
    }
}

#[stable(feature = "foo", since = "1.0")]
#[rustc_const_unstable(feature = "const_t_try", issue = "none")]
impl const FromResidual for T {
    fn from_residual(t: T) -> T {
        t
    }
}

#[stable(feature = "foo", since = "1.0")]
pub trait Tr {
    #[default_method_body_is_const]
    #[stable(feature = "foo", since = "1.0")]
    fn bar() -> T {
        T?
        // Should be allowed.
        // Must enable unstable features to call this trait fn in const contexts.
    }
}

fn main() {}
