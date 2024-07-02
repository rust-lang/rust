// Ensure that rustdoc gives errors for trait impls inside function bodies that don't resolve.
// See https://github.com/rust-lang/rust/pull/73566
pub struct ValidType;
pub trait ValidTrait {}
pub trait NeedsBody {
    type Item;
    fn f();
}

/// This function has docs
pub fn f<B: UnknownBound>(a: UnknownType, b: B) {
//~^ ERROR cannot find trait `UnknownBound`
//~| ERROR cannot find type `UnknownType`
    impl UnknownTrait for ValidType {} //~ ERROR cannot find trait `UnknownTrait`
    impl<T: UnknownBound> UnknownTrait for T {}
    //~^ ERROR cannot find trait `UnknownBound`
    //~| ERROR cannot find trait `UnknownTrait`
    impl ValidTrait for UnknownType {}
    //~^ ERROR cannot find type `UnknownType`
    impl ValidTrait for ValidType where ValidTrait: UnknownBound {}
    //~^ ERROR cannot find trait `UnknownBound`

    /// This impl has documentation
    impl NeedsBody for ValidType {
        type Item = UnknownType;
        //~^ ERROR cannot find type `UnknownType`

        /// This function has documentation
        fn f() {
            <UnknownTypeShouldBeIgnored>::a();
            content::shouldnt::matter();
            unknown_macro!();
            //~^ ERROR cannot find macro `unknown_macro`

            /// This is documentation for a macro
            macro_rules! can_define_macros_here_too {
                () => {
                    this::content::should::also::be::ignored()
                }
            }
            can_define_macros_here_too!();

            /// This also is documented.
            pub fn doubly_nested(c: UnknownType) {
            //~^ ERROR cannot find type `UnknownType`
            }
        }
    }
}
