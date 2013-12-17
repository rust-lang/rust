// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast aux-build
// aux-build:lint_stability.rs

#[feature(globs)];
#[deny(unstable)];
#[deny(deprecated)];
#[deny(experimental)];
#[allow(dead_code)];

mod cross_crate {
    extern mod lint_stability;
    use self::lint_stability::*;

    fn test() {
        // XXX: attributes on methods are not encoded cross crate.
        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text

        experimental(); //~ ERROR use of experimental item
        foo.method_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental(); //~ ERROR use of experimental item

        experimental_text(); //~ ERROR use of experimental item: text
        foo.method_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text

        unstable(); //~ ERROR use of unstable item
        foo.method_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable(); //~ ERROR use of unstable item

        unstable_text(); //~ ERROR use of unstable item: text
        foo.method_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text

        unmarked(); //~ ERROR use of unmarked item
        foo.method_unmarked(); //~ ERROR use of unmarked item
        foo.trait_unmarked(); //~ ERROR use of unmarked item

        stable();
        foo.method_stable();
        foo.trait_stable();

        stable_text();
        foo.method_stable_text();
        foo.trait_stable_text();

        frozen();
        foo.method_frozen();
        foo.trait_frozen();

        frozen_text();
        foo.method_frozen_text();
        foo.trait_frozen_text();

        locked();
        foo.method_locked();
        foo.trait_locked();

        locked_text();
        foo.method_locked_text();
        foo.trait_locked_text();


        let _ = DeprecatedStruct { i: 0 }; //~ ERROR use of deprecated item
        let _ = ExperimentalStruct { i: 0 }; //~ ERROR use of experimental item
        let _ = UnstableStruct { i: 0 }; //~ ERROR use of unstable item
        let _ = UnmarkedStruct { i: 0 }; //~ ERROR use of unmarked item
        let _ = StableStruct { i: 0 };
        let _ = FrozenStruct { i: 0 };
        let _ = LockedStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item
        let _ = ExperimentalUnitStruct; //~ ERROR use of experimental item
        let _ = UnstableUnitStruct; //~ ERROR use of unstable item
        let _ = UnmarkedUnitStruct; //~ ERROR use of unmarked item
        let _ = StableUnitStruct;
        let _ = FrozenUnitStruct;
        let _ = LockedUnitStruct;

        let _ = DeprecatedVariant; //~ ERROR use of deprecated item
        let _ = ExperimentalVariant; //~ ERROR use of experimental item
        let _ = UnstableVariant; //~ ERROR use of unstable item
        let _ = UnmarkedVariant; //~ ERROR use of unmarked item
        let _ = StableVariant;
        let _ = FrozenVariant;
        let _ = LockedVariant;
    }

    fn test_method_param<F: Trait>(foo: F) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }
}

mod this_crate {
    #[deprecated]
    pub fn deprecated() {}
    #[deprecated="text"]
    pub fn deprecated_text() {}

    #[experimental]
    pub fn experimental() {}
    #[experimental="text"]
    pub fn experimental_text() {}

    #[unstable]
    pub fn unstable() {}
    #[unstable="text"]
    pub fn unstable_text() {}

    pub fn unmarked() {}

    #[stable]
    pub fn stable() {}
    #[stable="text"]
    pub fn stable_text() {}

    #[locked]
    pub fn locked() {}
    #[locked="text"]
    pub fn locked_text() {}

    #[frozen]
    pub fn frozen() {}
    #[frozen="text"]
    pub fn frozen_text() {}

    #[stable]
    pub struct MethodTester;

    impl MethodTester {
        #[deprecated]
        pub fn method_deprecated(&self) {}
        #[deprecated="text"]
        pub fn method_deprecated_text(&self) {}

        #[experimental]
        pub fn method_experimental(&self) {}
        #[experimental="text"]
        pub fn method_experimental_text(&self) {}

        #[unstable]
        pub fn method_unstable(&self) {}
        #[unstable="text"]
        pub fn method_unstable_text(&self) {}

        pub fn method_unmarked(&self) {}

        #[stable]
        pub fn method_stable(&self) {}
        #[stable="text"]
        pub fn method_stable_text(&self) {}

        #[locked]
        pub fn method_locked(&self) {}
        #[locked="text"]
        pub fn method_locked_text(&self) {}

        #[frozen]
        pub fn method_frozen(&self) {}
        #[frozen="text"]
        pub fn method_frozen_text(&self) {}
    }

    pub trait Trait {
        #[deprecated]
        fn trait_deprecated(&self) {}
        #[deprecated="text"]
        fn trait_deprecated_text(&self) {}

        #[experimental]
        fn trait_experimental(&self) {}
        #[experimental="text"]
        fn trait_experimental_text(&self) {}

        #[unstable]
        fn trait_unstable(&self) {}
        #[unstable="text"]
        fn trait_unstable_text(&self) {}

        fn trait_unmarked(&self) {}

        #[stable]
        fn trait_stable(&self) {}
        #[stable="text"]
        fn trait_stable_text(&self) {}

        #[locked]
        fn trait_locked(&self) {}
        #[locked="text"]
        fn trait_locked_text(&self) {}

        #[frozen]
        fn trait_frozen(&self) {}
        #[frozen="text"]
        fn trait_frozen_text(&self) {}
    }

    impl Trait for MethodTester {}

    #[deprecated]
    pub struct DeprecatedStruct { i: int }
    #[experimental]
    pub struct ExperimentalStruct { i: int }
    #[unstable]
    pub struct UnstableStruct { i: int }
    pub struct UnmarkedStruct { i: int }
    #[stable]
    pub struct StableStruct { i: int }
    #[frozen]
    pub struct FrozenStruct { i: int }
    #[locked]
    pub struct LockedStruct { i: int }

    #[deprecated]
    pub struct DeprecatedUnitStruct;
    #[experimental]
    pub struct ExperimentalUnitStruct;
    #[unstable]
    pub struct UnstableUnitStruct;
    pub struct UnmarkedUnitStruct;
    #[stable]
    pub struct StableUnitStruct;
    #[frozen]
    pub struct FrozenUnitStruct;
    #[locked]
    pub struct LockedUnitStruct;

    pub enum Enum {
        #[deprecated]
        DeprecatedVariant,
        #[experimental]
        ExperimentalVariant,
        #[unstable]
        UnstableVariant,

        UnmarkedVariant,
        #[stable]
        StableVariant,
        #[frozen]
        FrozenVariant,
        #[locked]
        LockedVariant,
    }

    fn test() {
        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text

        experimental(); //~ ERROR use of experimental item
        foo.method_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental(); //~ ERROR use of experimental item

        experimental_text(); //~ ERROR use of experimental item: text
        foo.method_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text

        unstable(); //~ ERROR use of unstable item
        foo.method_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable(); //~ ERROR use of unstable item

        unstable_text(); //~ ERROR use of unstable item: text
        foo.method_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text

        unmarked(); //~ ERROR use of unmarked item
        foo.method_unmarked(); //~ ERROR use of unmarked item
        foo.trait_unmarked(); //~ ERROR use of unmarked item

        stable();
        foo.method_stable();
        foo.trait_stable();

        stable_text();
        foo.method_stable_text();
        foo.trait_stable_text();

        frozen();
        foo.method_frozen();
        foo.trait_frozen();

        frozen_text();
        foo.method_frozen_text();
        foo.trait_frozen_text();

        locked();
        foo.method_locked();
        foo.trait_locked();

        locked_text();
        foo.method_locked_text();
        foo.trait_locked_text();


        let _ = DeprecatedStruct { i: 0 }; //~ ERROR use of deprecated item
        let _ = ExperimentalStruct { i: 0 }; //~ ERROR use of experimental item
        let _ = UnstableStruct { i: 0 }; //~ ERROR use of unstable item
        let _ = UnmarkedStruct { i: 0 }; //~ ERROR use of unmarked item
        let _ = StableStruct { i: 0 };
        let _ = FrozenStruct { i: 0 };
        let _ = LockedStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item
        let _ = ExperimentalUnitStruct; //~ ERROR use of experimental item
        let _ = UnstableUnitStruct; //~ ERROR use of unstable item
        let _ = UnmarkedUnitStruct; //~ ERROR use of unmarked item
        let _ = StableUnitStruct;
        let _ = FrozenUnitStruct;
        let _ = LockedUnitStruct;

        let _ = DeprecatedVariant; //~ ERROR use of deprecated item
        let _ = ExperimentalVariant; //~ ERROR use of experimental item
        let _ = UnstableVariant; //~ ERROR use of unstable item
        let _ = UnmarkedVariant; //~ ERROR use of unmarked item
        let _ = StableVariant;
        let _ = FrozenVariant;
        let _ = LockedVariant;
    }

    fn test_method_param<F: Trait>(foo: F) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_experimental(); //~ ERROR use of experimental item
        foo.trait_experimental_text(); //~ ERROR use of experimental item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }
}

fn main() {}
