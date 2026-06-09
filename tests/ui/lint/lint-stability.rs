//@ aux-build:lint_stability.rs
//@ aux-build:inherited_stability.rs
//@ aux-build:stability_cfg1.rs
//@ aux-build:stability-cfg2.rs

#![allow(deprecated)]
#![allow(dead_code)]
#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

#[macro_use]
extern crate lint_stability;

mod cross_crate {
    extern crate stability_cfg1;
    extern crate stability_cfg2; //~ ERROR use of unstable library feature

    use lint_stability::*;

    fn test() {
        type Foo = MethodTester;
        let foo = MethodTester;

        deprecated();
        foo.method_deprecated();
        Foo::method_deprecated(&foo);
        <Foo>::method_deprecated(&foo);
        foo.trait_deprecated();
        Trait::trait_deprecated(&foo);
        <Foo>::trait_deprecated(&foo);
        <Foo as Trait>::trait_deprecated(&foo);

        deprecated_text();
        foo.method_deprecated_text();
        Foo::method_deprecated_text(&foo);
        <Foo>::method_deprecated_text(&foo);
        foo.trait_deprecated_text();
        Trait::trait_deprecated_text(&foo);
        <Foo>::trait_deprecated_text(&foo);
        <Foo as Trait>::trait_deprecated_text(&foo);

        deprecated_future(); // Fine; no error.

        deprecated_unstable();
        //~^ ERROR use of unstable library feature
        Trait::trait_deprecated_unstable(&foo);
        //~^ ERROR use of unstable library feature
        <Foo as Trait>::trait_deprecated_unstable(&foo);
        //~^ ERROR use of unstable library feature

        deprecated_unstable_text();
        //~^ ERROR use of unstable library feature
        Trait::trait_deprecated_unstable_text(&foo);
        //~^ ERROR use of unstable library feature
        <Foo as Trait>::trait_deprecated_unstable_text(&foo);
        //~^ ERROR use of unstable library feature

        unstable(); //~ ERROR use of unstable library feature
        Trait::trait_unstable(&foo); //~ ERROR use of unstable library feature
        <Foo as Trait>::trait_unstable(&foo); //~ ERROR use of unstable library feature

        unstable_text();
        //~^ ERROR use of unstable library feature `unstable_test_feature`: text
        Trait::trait_unstable_text(&foo);
        //~^ ERROR use of unstable library feature `unstable_test_feature`: text
        <Foo as Trait>::trait_unstable_text(&foo);
        //~^ ERROR use of unstable library feature `unstable_test_feature`: text

        stable();
        foo.method_stable();
        Foo::method_stable(&foo);
        <Foo>::method_stable(&foo);
        foo.trait_stable();
        Trait::trait_stable(&foo);
        <Foo>::trait_stable(&foo);
        <Foo as Trait>::trait_stable(&foo);

        stable_text();
        foo.method_stable_text();
        Foo::method_stable_text(&foo);
        <Foo>::method_stable_text(&foo);
        foo.trait_stable_text();
        Trait::trait_stable_text(&foo);
        <Foo>::trait_stable_text(&foo);
        <Foo as Trait>::trait_stable_text(&foo);

        struct S1<T: TraitWithAssociatedTypes>(T::TypeUnstable);
        //~^ ERROR use of unstable library feature
        struct S2<T: TraitWithAssociatedTypes>(T::TypeDeprecated);
        type A = dyn TraitWithAssociatedTypes<
            TypeUnstable = u8, //~ ERROR use of unstable library feature
            TypeDeprecated = u16,
        >;

        let _ = DeprecatedStruct {
            i: 0
        };
        let _ = DeprecatedUnstableStruct {
            //~^ ERROR use of unstable library feature
            i: 0
        };
        let _ = UnstableStruct { i: 0 }; //~ ERROR use of unstable library feature
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct;
        let _ = DeprecatedUnstableUnitStruct;
        //~^ ERROR use of unstable library feature
        let _ = UnstableUnitStruct; //~ ERROR use of unstable library feature
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant;
        let _ = Enum::DeprecatedUnstableVariant;
        //~^ ERROR use of unstable library feature
        let _ = Enum::UnstableVariant; //~ ERROR use of unstable library feature
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1);
        let _ = DeprecatedUnstableTupleStruct (1);
        //~^ ERROR use of unstable library feature
        let _ = UnstableTupleStruct (1); //~ ERROR use of unstable library feature
        let _ = StableTupleStruct (1);

        // At the moment, the lint checker only checks stability in
        // in the arguments of macros.
        // Eventually, we will want to lint the contents of the
        // macro in the module *defining* it. Also, stability levels
        // on macros themselves are not yet linted.
        macro_test_arg!(deprecated_text());
        macro_test_arg!(deprecated_unstable_text());
        //~^ ERROR use of unstable library feature
        macro_test_arg!(macro_test_arg!(deprecated_text()));
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated();
        Trait::trait_deprecated(&foo);
        <Foo>::trait_deprecated(&foo);
        <Foo as Trait>::trait_deprecated(&foo);
        foo.trait_deprecated_text();
        Trait::trait_deprecated_text(&foo);
        <Foo>::trait_deprecated_text(&foo);
        <Foo as Trait>::trait_deprecated_text(&foo);
        Trait::trait_deprecated_unstable(&foo);
        //~^ ERROR use of unstable library feature
        <Foo as Trait>::trait_deprecated_unstable(&foo);
        //~^ ERROR use of unstable library feature
        Trait::trait_deprecated_unstable_text(&foo);
        //~^ ERROR use of unstable library feature
        <Foo as Trait>::trait_deprecated_unstable_text(&foo);
        //~^ ERROR use of unstable library feature
        Trait::trait_unstable(&foo); //~ ERROR use of unstable library feature
        <Foo as Trait>::trait_unstable(&foo); //~ ERROR use of unstable library feature
        Trait::trait_unstable_text(&foo);
        //~^ ERROR use of unstable library feature `unstable_test_feature`: text
        <Foo as Trait>::trait_unstable_text(&foo);
        //~^ ERROR use of unstable library feature `unstable_test_feature`: text
        foo.trait_stable();
        Trait::trait_stable(&foo);
        <Foo>::trait_stable(&foo);
        <Foo as Trait>::trait_stable(&foo);
    }

    fn test_method_object(foo: &dyn Trait) {
        foo.trait_deprecated();
        foo.trait_deprecated_text();
        foo.trait_stable();
    }

    struct S;

    impl UnstableTrait for S { } //~ ERROR use of unstable library feature
    impl DeprecatedTrait for S {}
    trait LocalTrait : UnstableTrait { } //~ ERROR use of unstable library feature
    trait LocalTrait2 : DeprecatedTrait { }

    impl Trait for S {
        fn trait_stable(&self) {}
        fn trait_unstable(&self) {} //~ ERROR use of unstable library feature
    }
}

mod inheritance {
    extern crate inherited_stability; //~ ERROR use of unstable library feature
    use self::inherited_stability::*; //~ ERROR use of unstable library feature

    fn test_inheritance() {
        unstable(); //~ ERROR use of unstable library feature
        stable();

        stable_mod::unstable(); //~ ERROR use of unstable library feature
        stable_mod::stable();

        unstable_mod::deprecated(); //~ ERROR use of unstable library feature
        unstable_mod::unstable(); //~ ERROR use of unstable library feature

        let _ = Unstable::UnstableVariant; //~ ERROR use of unstable library feature
        let _ = Unstable::StableVariant; //~ ERROR use of unstable library feature

        let x: usize = 0;
        x.stable();
    }
}

mod this_crate {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn deprecated() {}
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn deprecated_text() {}

    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "99.99.99", note = "text")]
    pub fn deprecated_future() {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub fn unstable() {}
    #[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
    pub fn unstable_text() {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn stable() {}
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn stable_text() {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct MethodTester;

    impl MethodTester {
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        pub fn method_deprecated(&self) {}
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        pub fn method_deprecated_text(&self) {}

        #[unstable(feature = "unstable_test_feature", issue = "none")]
        pub fn method_unstable(&self) {}
        #[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
        pub fn method_unstable_text(&self) {}

        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn method_stable(&self) {}
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn method_stable_text(&self) {}
    }

    pub trait Trait {
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        fn trait_deprecated(&self) {}
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        fn trait_deprecated_text(&self) {}

        #[unstable(feature = "unstable_test_feature", issue = "none")]
        fn trait_unstable(&self) {}
        #[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
        fn trait_unstable_text(&self) {}

        #[stable(feature = "rust1", since = "1.0.0")]
        fn trait_stable(&self) {}
        #[stable(feature = "rust1", since = "1.0.0")]
        fn trait_stable_text(&self) {}
    }

    impl Trait for MethodTester {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedStruct {
        #[stable(feature = "stable_test_feature", since = "1.0.0")] i: isize
    }
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub struct UnstableStruct {
        #[stable(feature = "stable_test_feature", since = "1.0.0")] i: isize
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct StableStruct {
        #[stable(feature = "stable_test_feature", since = "1.0.0")] i: isize
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedUnitStruct;
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub struct UnstableUnitStruct;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct StableUnitStruct;

    pub enum Enum {
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        DeprecatedVariant,
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        UnstableVariant,

        #[stable(feature = "rust1", since = "1.0.0")]
        StableVariant,
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedTupleStruct(isize);
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub struct UnstableTupleStruct(isize);
    #[stable(feature = "rust1", since = "1.0.0")]
    pub struct StableTupleStruct(isize);

    fn test() {
        // Only the deprecated cases of the following should generate
        // errors, because other stability attributes now have meaning
        // only *across* crates, not within a single crate.

        type Foo = MethodTester;
        let foo = MethodTester;

        deprecated();
        foo.method_deprecated();
        Foo::method_deprecated(&foo);
        <Foo>::method_deprecated(&foo);
        foo.trait_deprecated();
        Trait::trait_deprecated(&foo);
        <Foo>::trait_deprecated(&foo);
        <Foo as Trait>::trait_deprecated(&foo);

        deprecated_text();
        foo.method_deprecated_text();
        Foo::method_deprecated_text(&foo);
        <Foo>::method_deprecated_text(&foo);
        foo.trait_deprecated_text();
        Trait::trait_deprecated_text(&foo);
        <Foo>::trait_deprecated_text(&foo);
        <Foo as Trait>::trait_deprecated_text(&foo);

        deprecated_future();

        unstable();
        foo.method_unstable();
        Foo::method_unstable(&foo);
        <Foo>::method_unstable(&foo);
        foo.trait_unstable();
        Trait::trait_unstable(&foo);
        <Foo>::trait_unstable(&foo);
        <Foo as Trait>::trait_unstable(&foo);

        unstable_text();
        foo.method_unstable_text();
        Foo::method_unstable_text(&foo);
        <Foo>::method_unstable_text(&foo);
        foo.trait_unstable_text();
        Trait::trait_unstable_text(&foo);
        <Foo>::trait_unstable_text(&foo);
        <Foo as Trait>::trait_unstable_text(&foo);

        stable();
        foo.method_stable();
        Foo::method_stable(&foo);
        <Foo>::method_stable(&foo);
        foo.trait_stable();
        Trait::trait_stable(&foo);
        <Foo>::trait_stable(&foo);
        <Foo as Trait>::trait_stable(&foo);

        stable_text();
        foo.method_stable_text();
        Foo::method_stable_text(&foo);
        <Foo>::method_stable_text(&foo);
        foo.trait_stable_text();
        Trait::trait_stable_text(&foo);
        <Foo>::trait_stable_text(&foo);
        <Foo as Trait>::trait_stable_text(&foo);

        let _ = DeprecatedStruct {
            i: 0
        };
        let _ = UnstableStruct { i: 0 };
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct;
        let _ = UnstableUnitStruct;
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant;
        let _ = Enum::UnstableVariant;
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1);
        let _ = UnstableTupleStruct (1);
        let _ = StableTupleStruct (1);
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated();
        Trait::trait_deprecated(&foo);
        <Foo>::trait_deprecated(&foo);
        <Foo as Trait>::trait_deprecated(&foo);
        foo.trait_deprecated_text();
        Trait::trait_deprecated_text(&foo);
        <Foo>::trait_deprecated_text(&foo);
        <Foo as Trait>::trait_deprecated_text(&foo);
        foo.trait_unstable();
        Trait::trait_unstable(&foo);
        <Foo>::trait_unstable(&foo);
        <Foo as Trait>::trait_unstable(&foo);
        foo.trait_unstable_text();
        Trait::trait_unstable_text(&foo);
        <Foo>::trait_unstable_text(&foo);
        <Foo as Trait>::trait_unstable_text(&foo);
        foo.trait_stable();
        Trait::trait_stable(&foo);
        <Foo>::trait_stable(&foo);
        <Foo as Trait>::trait_stable(&foo);
    }

    fn test_method_object(foo: &dyn Trait) {
        foo.trait_deprecated();
        foo.trait_deprecated_text();
        foo.trait_unstable();
        foo.trait_unstable_text();
        foo.trait_stable();
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    fn test_fn_body() {
        fn fn_in_body() {}
        fn_in_body();
    }

    impl MethodTester {
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        fn test_method_body(&self) {
            fn fn_in_body() {}
            fn_in_body();
        }
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub trait DeprecatedTrait {
        fn dummy(&self) { }
    }

    struct S;

    impl DeprecatedTrait for S { }

    trait LocalTrait : DeprecatedTrait { }
}

fn main() {}
