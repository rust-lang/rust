// skip-filecheck
//@ test-mir-pass: GVN
//@ only-64bit
//@ compile-flags: -Z mir-enable-passes=+Inline

// Regression for <https://github.com/rust-lang/rust/issues/127089>

struct Foo<T>(std::marker::PhantomData<T>);

impl<T> Foo<T> {
    const SENTINEL: *mut T = std::ptr::dangling_mut();

    fn cmp_ptr(a: *mut T) -> bool {
        std::ptr::eq(a, Self::SENTINEL)
    }
}

// EMIT_MIR gvn_ptr_eq_with_constant.main.GVN.diff
pub fn main() {
    Foo::<u8>::cmp_ptr(std::ptr::dangling_mut());
}
