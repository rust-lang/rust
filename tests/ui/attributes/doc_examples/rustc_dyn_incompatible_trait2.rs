//@run-fail
mod foreign_crate {
    /// # Safety requirements
    ///
    /// If this type also implements `SafeTrait`,
    /// then that implementation must always return `true`.
    pub unsafe trait UnsafeTrait {}
    unsafe impl<T: UnsafeTrait + ?Sized> UnsafeTrait for &T {}

    pub trait SafeTrait {
        fn returns_true(&self) -> bool;
    }
    impl<T: SafeTrait + ?Sized> SafeTrait for &T {
        fn returns_true(&self) -> bool {
            (*self).returns_true()
        }
    }

    impl SafeTrait for u8 {
        fn returns_true(&self) -> bool {
            true
        }
    }
    /// Safety: impl returns `true`.
    unsafe impl UnsafeTrait for u8 {}

    pub fn function(x: impl UnsafeTrait + SafeTrait) {
        // Can't panic, after all, `x: UnsafeTrait`
        // guarantees `returns_true` actually returns `true`
        assert!(x.returns_true());
    }
}

use foreign_crate::{SafeTrait, UnsafeTrait, function};

pub trait LocalTrait: UnsafeTrait {}
impl<T: UnsafeTrait> LocalTrait for T {}

// We can do this because `dyn LocalTrait` is a local type.
// But `LocalTrait: UnsafeTrait`, so `dyn LocalTrait: UnsafeTrait` holds,
// and we don't have to `unsafe impl` it.
impl SafeTrait for dyn LocalTrait {
    fn returns_true(&self) -> bool {
        false
    }
}

fn main() {
    let x = 42_u8;
    let y: &dyn LocalTrait = &x;
    function(y); // panics
}
