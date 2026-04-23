#![feature(impl_restriction, auto_traits, const_trait_impl)]
#![expect(incomplete_features)]

mod foo {
    pub impl(crate::foo) trait Baz {} //~ ERROR incorrect `impl` restriction
    pub impl(crate::foo) unsafe trait BazUnsafe {} //~ ERROR incorrect `impl` restriction
    pub impl(crate::foo) auto trait BazAuto {} //~ ERROR incorrect `impl` restriction
    pub impl(crate::foo) const trait BazConst {} //~ ERROR incorrect `impl` restriction
    pub impl(crate::foo) const unsafe trait BazConstUnsafe {} //~ ERROR incorrect `impl` restriction
    pub impl(crate::foo) unsafe auto trait BazUnsafeAuto {} //~ ERROR incorrect `impl` restriction

    // FIXME: The positioning of `impl(..)` may be confusing.
    // When users get the keyword order wrong, the compiler currently emits
    // a generic "unexpected token" error, which is not very helpful.
    // In the future, we could improve diagnostics by detecting misordered
    // keywords and suggesting the correct order.
    pub unsafe impl(crate::foo) trait BadOrder1 {} //~ ERROR expected one of `for`, `where`, or `{`, found keyword `trait`

    // FIXME: The following cases are not checked for now,
    // as the compiler aborts due to the previous syntax error in `BadOrder1`.
    // In the future, we could recover from such errors and continue compilation.
    pub auto impl(crate::foo) trait BadOrder2 {}
    pub const impl(crate::foo) trait BadOrder3 {}
    pub unsafe auto impl(crate::foo) trait BadOrder4 {}
    pub const unsafe impl(crate::foo) trait BadOrder5 {}
    pub unsafe impl(crate::foo) auto trait BadOrder6 {}
    pub const impl(crate::foo) unsafe trait BadOrder7 {}
}

fn main() {}
