//@ compile-flags: -Znext-solver=globally

// A bogus `impl Sized for T` (rejected with E0322) used to cause an ICE when
// the new trait solver processed it and made DSTs like `[T]` appear sized. This
// caused pointer layout to be computed as a thin-pointer (Scalar) instead of a
// fat-pointer (ScalarPair), and const evaluation of unsafe pointer arithmetic
// inside a `const {}` block would then ICE with:
//   "invalid immediate for given destination place: value ScalarPair(...) does
//    not match ABI Scalar(...)"

impl<'a, T: ?Sized> Sized for T {}
//~^ ERROR explicit impls for the `Sized` trait are not permitted
//~| ERROR type parameter `T` must be used as the type parameter for some local type

fn main() {
    const {
        unsafe {
            let value = [1, 2];
            let _ptr = value.as_ptr().add(2);
        }
    }
}
