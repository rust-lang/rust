// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:deprecation-lint.rs

#![feature(deprecated)]

#![deny(deprecated)]
#![allow(warnings)]

#[macro_use]
extern crate deprecation_lint;

mod cross_crate {
    use deprecation_lint::*;

    fn test() {
        type Foo = MethodTester;
        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        Foo::method_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::method_deprecated(&foo); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        Trait::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo as Trait>::trait_deprecated(&foo); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        Foo::method_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::method_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        Trait::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text

        let _ = DeprecatedStruct { //~ ERROR use of deprecated item
            i: 0 //~ ERROR use of deprecated item
        };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item

        let _ = Enum::DeprecatedVariant; //~ ERROR use of deprecated item

        let _ = DeprecatedTupleStruct (1); //~ ERROR use of deprecated item

        // At the moment, the lint checker only checks stability in
        // in the arguments of macros.
        // Eventually, we will want to lint the contents of the
        // macro in the module *defining* it. Also, stability levels
        // on macros themselves are not yet linted.
        macro_test_arg!(deprecated_text()); //~ ERROR use of deprecated item: text
        macro_test_arg!(macro_test_arg!(deprecated_text())); //~ ERROR use of deprecated item: text
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        Trait::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo as Trait>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        Trait::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
    }

    pub fn foo() {
        let x = Stable {
            override2: 3,
            //~^ ERROR use of deprecated item
        };

        let _ = x.override2;
        //~^ ERROR use of deprecated item

        let Stable {
            override2: _
            //~^ ERROR use of deprecated item
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3);

        let _ = x.2;
        //~^ ERROR use of deprecated item

        let Stable2(_,
                   _,
                   _)
            //~^ ERROR use of deprecated item
            = x;
        // all fine
        let Stable2(..) = x;

        let x = Deprecated {
            //~^ ERROR use of deprecated item
            inherit: 1,
            //~^ ERROR use of deprecated item
        };

        let _ = x.inherit;
        //~^ ERROR use of deprecated item

        let Deprecated {
            //~^ ERROR use of deprecated item
            inherit: _,
            //~^ ERROR use of deprecated item
        } = x;

        let Deprecated
            //~^ ERROR use of deprecated item
            { .. } = x;

        let x = Deprecated2(1, 2, 3);
        //~^ ERROR use of deprecated item

        let _ = x.0;
        //~^ ERROR use of deprecated item
        let _ = x.1;
        //~^ ERROR use of deprecated item
        let _ = x.2;
        //~^ ERROR use of deprecated item

        let Deprecated2
        //~^ ERROR use of deprecated item
            (_,
             //~^ ERROR use of deprecated item
             _,
             //~^ ERROR use of deprecated item
             _)
             //~^ ERROR use of deprecated item
            = x;
        let Deprecated2
        //~^ ERROR use of deprecated item
            // the patterns are all fine:
            (..) = x;
    }
}

mod inheritance {
    use deprecation_lint::*;

    fn test_inheritance() {
        deprecated_mod::deprecated(); //~ ERROR use of deprecated item
    }
}

mod this_crate {
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn deprecated() {}
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn deprecated_text() {}

    pub struct MethodTester;

    impl MethodTester {
        #[deprecated(since = "1.0.0", note = "text")]
        pub fn method_deprecated(&self) {}
        #[deprecated(since = "1.0.0", note = "text")]
        pub fn method_deprecated_text(&self) {}
    }

    pub trait Trait {
        #[deprecated(since = "1.0.0", note = "text")]
        fn trait_deprecated(&self) {}
        #[deprecated(since = "1.0.0", note = "text")]
        fn trait_deprecated_text(&self) {}
    }

    impl Trait for MethodTester {}

    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedStruct {
        i: isize
    }
    pub struct UnstableStruct {
        i: isize
    }
    pub struct StableStruct {
        i: isize
    }

    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedUnitStruct;

    pub enum Enum {
        #[deprecated(since = "1.0.0", note = "text")]
        DeprecatedVariant,
    }

    #[deprecated(since = "1.0.0", note = "text")]
    pub struct DeprecatedTupleStruct(isize);

    fn test() {
        // Only the deprecated cases of the following should generate
        // errors, because other stability attributes now have meaning
        // only *across* crates, not within a single crate.

        type Foo = MethodTester;
        let foo = MethodTester;

        deprecated(); //~ ERROR use of deprecated item
        foo.method_deprecated(); //~ ERROR use of deprecated item
        Foo::method_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::method_deprecated(&foo); //~ ERROR use of deprecated item
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        Trait::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo as Trait>::trait_deprecated(&foo); //~ ERROR use of deprecated item

        deprecated_text(); //~ ERROR use of deprecated item: text
        foo.method_deprecated_text(); //~ ERROR use of deprecated item: text
        Foo::method_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::method_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        Trait::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text

        let _ = DeprecatedStruct {
            //~^ ERROR use of deprecated item
            i: 0 //~ ERROR use of deprecated item
        };

        let _ = DeprecatedUnitStruct; //~ ERROR use of deprecated item

        let _ = Enum::DeprecatedVariant; //~ ERROR use of deprecated item

        let _ = DeprecatedTupleStruct (1); //~ ERROR use of deprecated item
    }

    fn test_method_param<Foo: Trait>(foo: Foo) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        Trait::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        <Foo as Trait>::trait_deprecated(&foo); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
        Trait::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
        <Foo as Trait>::trait_deprecated_text(&foo); //~ ERROR use of deprecated item: text
    }

    fn test_method_object(foo: &Trait) {
        foo.trait_deprecated(); //~ ERROR use of deprecated item
        foo.trait_deprecated_text(); //~ ERROR use of deprecated item: text
    }

    #[deprecated(since = "1.0.0", note = "text")]
    fn test_fn_body() {
        fn fn_in_body() {}
        fn_in_body(); //~ ERROR use of deprecated item: text
    }

    impl MethodTester {
        #[deprecated(since = "1.0.0", note = "text")]
        fn test_method_body(&self) {
            fn fn_in_body() {}
            fn_in_body(); //~ ERROR use of deprecated item: text
        }
    }

    #[deprecated(since = "1.0.0", note = "text")]
    pub trait DeprecatedTrait {
        fn dummy(&self) { }
    }

    struct S;

    impl DeprecatedTrait for S { } //~ ERROR use of deprecated item

    trait LocalTrait : DeprecatedTrait { } //~ ERROR use of deprecated item
}

mod this_crate2 {
    struct Stable {
        #[deprecated(since = "1.0.0", note = "text")]
        override2: u8,
    }

    struct Stable2(u8,
                   u8,
                   #[deprecated(since = "1.0.0", note = "text")] u8);

    #[deprecated(since = "1.0.0", note = "text")]
    struct Deprecated {
        inherit: u8,
    }

    #[deprecated(since = "1.0.0", note = "text")]
    struct Deprecated2(u8,
                       u8,
                       u8);

    pub fn foo() {
        let x = Stable {
            override2: 3,
            //~^ ERROR use of deprecated item
        };

        let _ = x.override2;
        //~^ ERROR use of deprecated item

        let Stable {
            override2: _
            //~^ ERROR use of deprecated item
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3);

        let _ = x.2;
        //~^ ERROR use of deprecated item

        let Stable2(_,
                   _,
                   _)
            //~^ ERROR use of deprecated item
            = x;
        // all fine
        let Stable2(..) = x;

        let x = Deprecated {
            //~^ ERROR use of deprecated item
            inherit: 1,
            //~^ ERROR use of deprecated item
        };

        let _ = x.inherit;
        //~^ ERROR use of deprecated item

        let Deprecated {
            //~^ ERROR use of deprecated item
            inherit: _,
            //~^ ERROR use of deprecated item
        } = x;

        let Deprecated
            //~^ ERROR use of deprecated item
            // the patterns are all fine:
            { .. } = x;

        let x = Deprecated2(1, 2, 3);
        //~^ ERROR use of deprecated item

        let _ = x.0;
        //~^ ERROR use of deprecated item
        let _ = x.1;
        //~^ ERROR use of deprecated item
        let _ = x.2;
        //~^ ERROR use of deprecated item

        let Deprecated2
        //~^ ERROR use of deprecated item
            (_,
             //~^ ERROR use of deprecated item
             _,
             //~^ ERROR use of deprecated item
             _)
            //~^ ERROR use of deprecated item
            = x;
        let Deprecated2
        //~^ ERROR use of deprecated item
            // the patterns are all fine:
            (..) = x;
    }
}

fn main() {}
