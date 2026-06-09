//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

pub trait Trait: Sized {
    fn without_self() -> u32;
    fn without_self_default() -> u32 {
        0
    }

    fn with_default_impl(self) -> Self {
        self
    }
    fn with_default_impl_generic<T>(self, x: T) -> (Self, T) {
        (self, x)
    }

    fn without_default_impl(x: u32) -> (Self, u32);
    fn without_default_impl_generic<T>(x: T) -> (Self, T);
}

impl Trait for char {
    fn without_self() -> u32 {
        2
    }
    fn without_default_impl(x: u32) -> (Self, u32) {
        ('c', x)
    }
    fn without_default_impl_generic<T>(x: T) -> (Self, T) {
        ('c', x)
    }
}

impl Trait for u32 {
    fn without_self() -> u32 {
        1
    }
    fn without_default_impl(x: u32) -> (Self, u32) {
        (0, x)
    }
    fn without_default_impl_generic<T>(x: T) -> (Self, T) {
        (0, x)
    }
}
