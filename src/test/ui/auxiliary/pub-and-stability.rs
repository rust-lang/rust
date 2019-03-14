// This crate attempts to enumerate the various scenarios for how a
// type can define fields and methods with various visibilities and
// stabilities.
//
// The basic stability pattern in this file has four cases:
// 1. no stability attribute at all
// 2. a stable attribute (feature "unit_test")
// 3. an unstable attribute that unit test declares (feature "unstable_declared")
// 4. an unstable attribute that unit test fails to declare (feature "unstable_undeclared")
//
// This file also covers four kinds of visibility: private,
// pub(module), pub(crate), and pub.
//
// However, since stability attributes can only be observed in
// cross-crate linkage scenarios, there is little reason to take the
// cross-product (4 stability cases * 4 visibility cases), because the
// first three visibility cases cannot be accessed outside this crate,
// and therefore stability is only relevant when the visibility is pub
// to the whole universe.
//
// (The only reason to do so would be if one were worried about the
// compiler having some subtle bug where adding a stability attribute
// introduces a privacy violation. As a way to provide evidence that
// this is not occurring, I have put stability attributes on some
// non-pub fields, marked with SILLY below)

#![feature(staged_api)]

#![stable(feature = "unit_test", since = "1.0.0")]

#[stable(feature = "unit_test", since = "1.0.0")]
pub use m::{Record, Trait, Tuple};

mod m {
    #[derive(Default)]
    #[stable(feature = "unit_test", since = "1.0.0")]
    pub struct Record {
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub a_stable_pub: i32,
        #[unstable(feature = "unstable_declared", issue = "38412")]
        pub a_unstable_declared_pub: i32,
        #[unstable(feature = "unstable_undeclared", issue = "38412")]
        pub a_unstable_undeclared_pub: i32,
        #[unstable(feature = "unstable_undeclared", issue = "38412")] // SILLY
        pub(crate) b_crate: i32,
        #[unstable(feature = "unstable_declared", issue = "38412")] // SILLY
        pub(in m) c_mod: i32,
        #[stable(feature = "unit_test", since = "1.0.0")] // SILLY
        d_priv: i32
    }

    #[derive(Default)]
    #[stable(feature = "unit_test", since = "1.0.0")]
    pub struct Tuple(
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub i32,
        #[unstable(feature = "unstable_declared", issue = "38412")]
        pub i32,
        #[unstable(feature = "unstable_undeclared", issue = "38412")]
        pub i32,

        pub(crate) i32,
        pub(in m) i32,
        i32);

    impl Record {
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub fn new() -> Self { Default::default() }
    }

    impl Tuple {
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub fn new() -> Self { Default::default() }
    }


    #[stable(feature = "unit_test", since = "1.0.0")]
    pub trait Trait {
        #[stable(feature = "unit_test", since = "1.0.0")]
        type Type;
        #[stable(feature = "unit_test", since = "1.0.0")]
        fn stable_trait_method(&self) -> Self::Type;
        #[unstable(feature = "unstable_undeclared", issue = "38412")]
        fn unstable_undeclared_trait_method(&self) -> Self::Type;
        #[unstable(feature = "unstable_declared", issue = "38412")]
        fn unstable_declared_trait_method(&self) -> Self::Type;
    }

    #[stable(feature = "unit_test", since = "1.0.0")]
    impl Trait for Record {
        type Type = i32;
        fn stable_trait_method(&self) -> i32 { self.d_priv }
        fn unstable_undeclared_trait_method(&self) -> i32 { self.d_priv }
        fn unstable_declared_trait_method(&self) -> i32 { self.d_priv }
    }

    #[stable(feature = "unit_test", since = "1.0.0")]
    impl Trait for Tuple {
        type Type = i32;
        fn stable_trait_method(&self) -> i32 { self.3 }
        fn unstable_undeclared_trait_method(&self) -> i32 { self.3 }
        fn unstable_declared_trait_method(&self) -> i32 { self.3 }
    }

    impl Record {
        #[unstable(feature = "unstable_undeclared", issue = "38412")]
        pub fn unstable_undeclared(&self) -> i32 { self.d_priv }
        #[unstable(feature = "unstable_declared", issue = "38412")]
        pub fn unstable_declared(&self) -> i32 { self.d_priv }
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub fn stable(&self) -> i32 { self.d_priv }

        #[unstable(feature = "unstable_undeclared", issue = "38412")] // SILLY
        pub(crate) fn pub_crate(&self) -> i32 { self.d_priv }
        #[unstable(feature = "unstable_declared", issue = "38412")] // SILLY
        pub(in m) fn pub_mod(&self) -> i32 { self.d_priv }
        #[stable(feature = "unit_test", since = "1.0.0")] // SILLY
        fn private(&self) -> i32 { self.d_priv }
    }

    impl Tuple {
        #[unstable(feature = "unstable_undeclared", issue = "38412")]
        pub fn unstable_undeclared(&self) -> i32 { self.0 }
        #[unstable(feature = "unstable_declared", issue = "38412")]
        pub fn unstable_declared(&self) -> i32 { self.0 }
        #[stable(feature = "unit_test", since = "1.0.0")]
        pub fn stable(&self) -> i32 { self.0 }

        pub(crate) fn pub_crate(&self) -> i32 { self.0 }
        pub(in m) fn pub_mod(&self) -> i32 { self.0 }
        fn private(&self) -> i32 { self.0 }
    }
}
