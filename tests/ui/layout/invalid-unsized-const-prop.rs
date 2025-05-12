// issue: #127737
//@ check-pass
//@ compile-flags: -Zmir-opt-level=5 --crate-type lib

//! This test is very similar to `invalid-unsized-const-eval.rs`, but also requires
//! checking for unsized types in the last field of each enum variant.

pub trait TestTrait {
    type MyType;
    fn func() -> Option<Self>
    where
        Self: Sized;
}

impl<T> dyn TestTrait<MyType = T>
where
    Self: Sized,
{
    pub fn other_func() -> Option<Self> {
        match Self::func() {
            Some(me) => Some(me),
            None => None,
        }
    }
}
