// check-pass
// aux-build:lint_stability.rs
// aux-build:inherited_stability.rs
// aux-build:stability_cfg1.rs
// aux-build:stability-cfg2.rs
#![warn(deprecated)]
#![feature(staged_api, unstable_test_feature)]

#![stable(feature = "rust1", since = "1.0.0")]

#[macro_use]
extern crate lint_stability;

mod cross_crate {
    extern crate stability_cfg1;
    extern crate stability_cfg2;

    use lint_stability::*;

    fn test() {
        type Foo = MethodTester;
        let foo = MethodTester;

        deprecated(); //~ WARN use of deprecated function `lint_stability::deprecated`
        foo.method_deprecated(); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated`
        Foo::method_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated`
        <Foo>::method_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated`
        foo.trait_deprecated(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        Trait::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        <Foo>::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        <Foo as Trait>::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`

        deprecated_text(); //~ WARN use of deprecated function `lint_stability::deprecated_text`: text
        foo.method_deprecated_text(); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_text`: text
        Foo::method_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_text`: text
        <Foo>::method_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_text`: text
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        Trait::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        <Foo>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text

        deprecated_unstable(); //~ WARN use of deprecated function `lint_stability::deprecated_unstable`
        foo.method_deprecated_unstable(); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable`
        Foo::method_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable`
        <Foo>::method_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable`
        foo.trait_deprecated_unstable(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        Trait::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        <Foo>::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        <Foo as Trait>::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`

        deprecated_unstable_text(); //~ WARN use of deprecated function `lint_stability::deprecated_unstable_text`: text
        foo.method_deprecated_unstable_text(); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable_text`: text
        Foo::method_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable_text`: text
        <Foo>::method_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::MethodTester::method_deprecated_unstable_text`: text
        foo.trait_deprecated_unstable_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        Trait::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        <Foo>::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        <Foo as Trait>::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text

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

        struct S1<T: TraitWithAssociatedTypes>(T::TypeUnstable);
        struct S2<T: TraitWithAssociatedTypes>(T::TypeDeprecated);
        //~^ WARN use of deprecated associated type `lint_stability::TraitWithAssociatedTypes::TypeDeprecated`: text
        //~| WARN use of deprecated associated type `lint_stability::TraitWithAssociatedTypes::TypeDeprecated`: text
        type A = dyn TraitWithAssociatedTypes<
            TypeUnstable = u8,
            TypeDeprecated = u16,
            //~^ WARN use of deprecated associated type `lint_stability::TraitWithAssociatedTypes::TypeDeprecated`
            //~| WARN use of deprecated associated type `lint_stability::TraitWithAssociatedTypes::TypeDeprecated`
            //~| WARN use of deprecated associated type `lint_stability::TraitWithAssociatedTypes::TypeDeprecated`
        >;

        let _ = DeprecatedStruct { //~ WARN use of deprecated struct `lint_stability::DeprecatedStruct`
            i: 0 //~ WARN use of deprecated field `lint_stability::DeprecatedStruct::i`
        };
        let _ = DeprecatedUnstableStruct {
            //~^ WARN use of deprecated struct `lint_stability::DeprecatedUnstableStruct`
            i: 0 //~ WARN use of deprecated field `lint_stability::DeprecatedUnstableStruct::i`
        };
        let _ = UnstableStruct { i: 0 };
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ WARN use of deprecated unit struct `lint_stability::DeprecatedUnitStruct`
        let _ = DeprecatedUnstableUnitStruct; //~ WARN use of deprecated unit struct `lint_stability::DeprecatedUnstableUnitStruct`
        let _ = UnstableUnitStruct;
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant; //~ WARN use of deprecated unit variant `lint_stability::Enum::DeprecatedVariant`
        let _ = Enum::DeprecatedUnstableVariant; //~ WARN use of deprecated unit variant `lint_stability::Enum::DeprecatedUnstableVariant`
        let _ = Enum::UnstableVariant;
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1); //~ WARN use of deprecated tuple struct `lint_stability::DeprecatedTupleStruct`
        let _ = DeprecatedUnstableTupleStruct (1); //~ WARN use of deprecated tuple struct `lint_stability::DeprecatedUnstableTupleStruct`
        let _ = UnstableTupleStruct (1);
        let _ = StableTupleStruct (1);

        // At the moment, the lint checker only checks stability
        // in the arguments of macros.
        // Eventually, we will want to lint the contents of the
        // macro in the module *defining* it. Also, stability levels
        // on macros themselves are not yet linted.
        macro_test_arg!(deprecated_text()); //~ WARN use of deprecated function `lint_stability::deprecated_text`: text
        macro_test_arg!(deprecated_unstable_text()); //~ WARN use of deprecated function `lint_stability::deprecated_unstable_text`: text
        macro_test_arg!(macro_test_arg!(deprecated_text())); //~ WARN use of deprecated function `lint_stability::deprecated_text`: text
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        Trait::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        <Foo>::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        <Foo as Trait>::trait_deprecated(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        Trait::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        <Foo>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        foo.trait_deprecated_unstable(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        Trait::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        <Foo>::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        <Foo as Trait>::trait_deprecated_unstable(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        foo.trait_deprecated_unstable_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        Trait::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        <Foo>::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        <Foo as Trait>::trait_deprecated_unstable_text(&foo); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
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
        foo.trait_deprecated(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated`
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_text`: text
        foo.trait_deprecated_unstable(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable`
        foo.trait_deprecated_unstable_text(); //~ WARN use of deprecated method `lint_stability::Trait::trait_deprecated_unstable_text`: text
        foo.trait_unstable();
        foo.trait_unstable_text();
        foo.trait_stable();
    }

    struct S;

    impl UnstableTrait for S { }
    impl DeprecatedTrait for S {} //~ WARN use of deprecated trait `lint_stability::DeprecatedTrait`: text
    trait LocalTrait : UnstableTrait { }
    trait LocalTrait2 : DeprecatedTrait { } //~ WARN use of deprecated trait `lint_stability::DeprecatedTrait`: text

    impl Trait for S {
        fn trait_stable(&self) {}
        fn trait_unstable(&self) {}
    }
}

mod inheritance {
    extern crate inherited_stability;
    use self::inherited_stability::*;

    fn test_inheritance() {
        unstable();
        stable();

        stable_mod::unstable();
        stable_mod::stable();

        unstable_mod::deprecated(); //~ WARN use of deprecated function `inheritance::inherited_stability::unstable_mod::deprecated`: text
        unstable_mod::unstable();

        let _ = Unstable::UnstableVariant;
        let _ = Unstable::StableVariant;

        let x: usize = 0;
        x.unstable();
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

        deprecated(); //~ WARN use of deprecated function `this_crate::deprecated`
        foo.method_deprecated(); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated`
        Foo::method_deprecated(&foo); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated`
        <Foo>::method_deprecated(&foo); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated`
        foo.trait_deprecated(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        Trait::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        <Foo>::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        <Foo as Trait>::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`

        deprecated_text(); //~ WARN use of deprecated function `this_crate::deprecated_text`: text
        foo.method_deprecated_text(); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated_text`: text
        Foo::method_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated_text`: text
        <Foo>::method_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::MethodTester::method_deprecated_text`: text
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        Trait::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        <Foo>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text

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
            //~^ WARN use of deprecated struct `this_crate::DeprecatedStruct`
            i: 0 //~ WARN use of deprecated field `this_crate::DeprecatedStruct::i`
        };
        let _ = UnstableStruct { i: 0 };
        let _ = StableStruct { i: 0 };

        let _ = DeprecatedUnitStruct; //~ WARN use of deprecated unit struct `this_crate::DeprecatedUnitStruct`
        let _ = UnstableUnitStruct;
        let _ = StableUnitStruct;

        let _ = Enum::DeprecatedVariant; //~ WARN use of deprecated unit variant `this_crate::Enum::DeprecatedVariant`
        let _ = Enum::UnstableVariant;
        let _ = Enum::StableVariant;

        let _ = DeprecatedTupleStruct (1); //~ WARN use of deprecated tuple struct `this_crate::DeprecatedTupleStruct`
        let _ = UnstableTupleStruct (1);
        let _ = StableTupleStruct (1);
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        Trait::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        <Foo>::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        <Foo as Trait>::trait_deprecated(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        Trait::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        <Foo>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
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
        foo.trait_deprecated(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated`
        foo.trait_deprecated_text(); //~ WARN use of deprecated method `this_crate::Trait::trait_deprecated_text`: text
        foo.trait_unstable();
        foo.trait_unstable_text();
        foo.trait_stable();
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    fn test_fn_body() {
        fn fn_in_body() {}
        fn_in_body(); //~ WARN use of deprecated function `this_crate::test_fn_body::fn_in_body`: text
    }

    impl MethodTester {
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        #[deprecated(since = "1.0.0", note = "text")]
        fn test_method_body(&self) {
            fn fn_in_body() {}
            fn_in_body(); //~ WARN use of deprecated function `this_crate::MethodTester::test_method_body::fn_in_body`: text
        }
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    pub trait DeprecatedTrait {
        fn dummy(&self) { }
    }

    struct S;

    impl DeprecatedTrait for S { } //~ WARN use of deprecated trait `this_crate::DeprecatedTrait`

    trait LocalTrait : DeprecatedTrait { } //~ WARN use of deprecated trait `this_crate::DeprecatedTrait`
}

fn main() {}
