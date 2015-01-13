// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_stability.rs
// aux-build:inherited_stability.rs
// aux-build:stability_cfg1.rs
// aux-build:stability_cfg2.rs

#![deny(unstable)]
#![deny(deprecated)]
#![deny(unstable)]
#![allow(dead_code)]
#![staged_api]

#[macro_use]
extern crate lint_stability; //~ ERROR: use of unmarked item

mod cross_crate {
    extern crate stability_cfg1;
    extern crate stability_cfg2; //~ ERROR: use of unstable item

    use lint_stability::*;

    fn test() {
        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text

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

        let _ = DeprecatedStruct { i: 0 }; //~ ERROR use of deprecated item
        let _ = UnstableStruct { i: 0 }; //~ ERROR use of unstable item
        let _ = UnmarkedStruct { i: 0 }; //~ ERROR use of unmarked item
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item
        let _ = UnstableUnitStruct; //~ ERROR use of unstable item
        let _ = UnmarkedUnitStruct; //~ ERROR use of unmarked item
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant; //~ ERROR use of deprecated item
        let _ = Enum::UnstableVariant; //~ ERROR use of unstable item
        let _ = Enum::UnmarkedVariant; //~ ERROR use of unmarked item
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1); //~ ERROR use of deprecated item
        let _ = UnstableTupleStruct (1); //~ ERROR use of unstable item
        let _ = UnmarkedTupleStruct (1); //~ ERROR use of unmarked item
        let _ = StableTupleStruct (1);

        // At the moment, the lint checker only checks stability in
        // in the arguments of macros.
        // Eventually, we will want to lint the contents of the
        // macro in the module *defining* it. Also, stability levels
        // on macros themselves are not yet linted.
        macro_test!();
        macro_test_arg!(deprecated_text()); //~ ERROR use of deprecated item: text
        macro_test_arg!(macro_test_arg!(deprecated_text())); //~ ERROR use of deprecated item: text
        macro_test_arg_nested!(deprecated_text);
    }

    fn test_method_param<F: Trait>(foo: F) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_unstable(); //~ ERROR use of unstable item
        foo.trait_unstable_text(); //~ ERROR use of unstable item: text
        foo.trait_unmarked(); //~ ERROR use of unmarked item
        foo.trait_stable();
    }

    struct S;

    impl UnstableTrait for S { } //~ ERROR use of unstable item

    trait LocalTrait : UnstableTrait { } //~ ERROR use of unstable item
}

mod inheritance {
    extern crate inherited_stability; //~ ERROR: use of unstable item
    use self::inherited_stability::*;

    fn test_inheritance() {
        unstable(); //~ ERROR use of unstable item
        stable();

        stable_mod::unstable(); //~ ERROR use of unstable item
        stable_mod::stable();

        unstable_mod::deprecated(); //~ ERROR use of deprecated item
        unstable_mod::unstable(); //~ ERROR use of unstable item

        let _ = Unstable::UnstableVariant; //~ ERROR use of unstable item
        let _ = Unstable::StableVariant;

        let x: usize = 0;
        x.unstable(); //~ ERROR use of unstable item
        x.stable();
    }
}

mod this_crate {
    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub fn deprecated() {}
    #[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
    pub fn deprecated_text() {}

    #[unstable(feature = "unnamed_feature", since = "1.0.0")]
    pub fn unstable() {}
    #[unstable(feature = "unnamed_feature", since = "1.0.0", reason = "text")]
    pub fn unstable_text() {}

    pub fn unmarked() {}

    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub fn stable() {}
    #[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
    pub fn stable_text() {}

    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub struct MethodTester;

    impl MethodTester {
        #[deprecated(feature = "oldstuff", since = "1.0.0")]
        pub fn method_deprecated(&self) {}
        #[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
        pub fn method_deprecated_text(&self) {}

        #[unstable(feature = "unnamed_feature", since = "1.0.0")]
        pub fn method_unstable(&self) {}
        #[unstable(feature = "unnamed_feature", since = "1.0.0", reason = "text")]
        pub fn method_unstable_text(&self) {}

        pub fn method_unmarked(&self) {}

        #[stable(feature = "grandfathered", since = "1.0.0")]
        pub fn method_stable(&self) {}
        #[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
        pub fn method_stable_text(&self) {}
    }

    pub trait Trait {
        #[deprecated(feature = "oldstuff", since = "1.0.0")]
        fn trait_deprecated(&self) {}
        #[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
        fn trait_deprecated_text(&self) {}

        #[unstable(feature = "unnamed_feature", since = "1.0.0")]
        fn trait_unstable(&self) {}
        #[unstable(feature = "unnamed_feature", since = "1.0.0", reason = "text")]
        fn trait_unstable_text(&self) {}

        fn trait_unmarked(&self) {}

        #[stable(feature = "grandfathered", since = "1.0.0")]
        fn trait_stable(&self) {}
        #[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
        fn trait_stable_text(&self) {}
    }

    impl Trait for MethodTester {}

    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub struct DeprecatedStruct { i: isize }
    #[unstable(feature = "unnamed_feature", since = "1.0.0")]
    pub struct UnstableStruct { i: isize }
    pub struct UnmarkedStruct { i: isize }
    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub struct StableStruct { i: isize }

    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub struct DeprecatedUnitStruct;
    #[unstable(feature = "unnamed_feature", since = "1.0.0")]
    pub struct UnstableUnitStruct;
    pub struct UnmarkedUnitStruct;
    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub struct StableUnitStruct;

    pub enum Enum {
        #[deprecated(feature = "oldstuff", since = "1.0.0")]
        DeprecatedVariant,
        #[unstable(feature = "unnamed_feature", since = "1.0.0")]
        UnstableVariant,

        UnmarkedVariant,
        #[stable(feature = "grandfathered", since = "1.0.0")]
        StableVariant,
    }

    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub struct DeprecatedTupleStruct(isize);
    #[unstable(feature = "unnamed_feature", since = "1.0.0")]
    pub struct UnstableTupleStruct(isize);
    pub struct UnmarkedTupleStruct(isize);
    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub struct StableTupleStruct(isize);

    fn test() {
        // Only the deprecated cases of the following should generate
        // errors, because other stability attributes now have meaning
        // only *across* crates, not within a single crate.

        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text

        unstable();
        foo.method_unstable();
        foo.trait_unstable();

        unstable_text();
        foo.method_unstable_text();
        foo.trait_unstable_text();

        unmarked();
        foo.method_unmarked();
        foo.trait_unmarked();

        stable();
        foo.method_stable();
        foo.trait_stable();

        stable_text();
        foo.method_stable_text();
        foo.trait_stable_text();

        let _ = DeprecatedStruct { i: 0 }; //~ ERROR use of deprecated item
        let _ = UnstableStruct { i: 0 };
        let _ = UnmarkedStruct { i: 0 };
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item
        let _ = UnstableUnitStruct;
        let _ = UnmarkedUnitStruct;
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant; //~ ERROR use of deprecated item
        let _ = Enum::UnstableVariant;
        let _ = Enum::UnmarkedVariant;
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1); //~ ERROR use of deprecated item
        let _ = UnstableTupleStruct (1);
        let _ = UnmarkedTupleStruct (1);
        let _ = StableTupleStruct (1);
    }

    fn test_method_param<F: Trait>(foo: F) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_unstable();
        foo.trait_unstable_text();
        foo.trait_unmarked();
        foo.trait_stable();
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        foo.trait_unstable();
        foo.trait_unstable_text();
        foo.trait_unmarked();
        foo.trait_stable();
    }

    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    fn test_fn_body() {
        fn fn_in_body() {}
        fn_in_body();
    }

    impl MethodTester {
        #[deprecated(feature = "oldstuff", since = "1.0.0")]
        fn test_method_body(&self) {
            fn fn_in_body() {}
            fn_in_body();
        }
    }

    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub trait DeprecatedTrait {}

    struct S;

    impl DeprecatedTrait for S { } //~ ERROR use of deprecated item

    trait LocalTrait : DeprecatedTrait { } //~ ERROR use of deprecated item
}

fn main() {}
