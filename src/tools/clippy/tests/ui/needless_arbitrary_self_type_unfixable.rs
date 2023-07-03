//@aux-build:proc_macro_attr.rs:proc-macro

#![warn(clippy::needless_arbitrary_self_type)]

#[macro_use]
extern crate proc_macro_attr;

mod issue_6089 {
    // Check that we don't lint if the `self` parameter comes from expansion

    macro_rules! test_from_expansion {
        () => {
            trait T1 {
                fn test(self: &Self);
            }

            struct S1;

            impl T1 for S1 {
                fn test(self: &Self) {}
            }
        };
    }

    test_from_expansion!();

    // If only the lifetime name comes from expansion we will lint, but the suggestion will have
    // placeholders and will not be applied automatically, as we can't reliably know the original name.
    // This specific case happened with async_trait.

    trait T2 {
        fn call_with_mut_self(&mut self);
    }

    struct S2;

    // The method's signature will be expanded to:
    //  fn call_with_mut_self<'life0>(self: &'life0 mut Self) {}
    #[rename_my_lifetimes]
    impl T2 for S2 {
        #[allow(clippy::needless_lifetimes)]
        fn call_with_mut_self(self: &mut Self) {}
    }
}

fn main() {}
