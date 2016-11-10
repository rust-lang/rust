// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_stability_fields.rs
#![allow(deprecated)]
#![allow(dead_code)]
#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod cross_crate {
    extern crate lint_stability_fields;

    use self::lint_stability_fields::*;

    pub fn foo() {
        let x = Stable {
            inherit: 1,
            override1: 2, //~ ERROR use of unstable
            override2: 3, //~ ERROR use of unstable
        };

        let _ = x.inherit;
        let _ = x.override1; //~ ERROR use of unstable
        let _ = x.override2; //~ ERROR use of unstable

        let Stable {
            inherit: _,
            override1: _, //~ ERROR use of unstable
            override2: _ //~ ERROR use of unstable
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1; //~ ERROR use of unstable
        let _ = x.2; //~ ERROR use of unstable

        let Stable2(_,
                   _, //~ ERROR use of unstable
                   _) //~ ERROR use of unstable
            = x;
        // all fine
        let Stable2(..) = x;


        let x = Unstable { //~ ERROR use of unstable
            inherit: 1, //~ ERROR use of unstable
            override1: 2,
            override2: 3, //~ ERROR use of unstable
        };

        let _ = x.inherit; //~ ERROR use of unstable
        let _ = x.override1;
        let _ = x.override2; //~ ERROR use of unstable

        let Unstable { //~ ERROR use of unstable
            inherit: _, //~ ERROR use of unstable
            override1: _,
            override2: _ //~ ERROR use of unstable
        } = x;

        let Unstable  //~ ERROR use of unstable
            // the patterns are all fine:
            { .. } = x;


        let x = Unstable2(1, 2, 3); //~ ERROR use of unstable

        let _ = x.0; //~ ERROR use of unstable
        let _ = x.1;
        let _ = x.2; //~ ERROR use of unstable

        let Unstable2  //~ ERROR use of unstable
            (_, //~ ERROR use of unstable
             _,
             _) //~ ERROR use of unstable
            = x;
        let Unstable2 //~ ERROR use of unstable
            // the patterns are all fine:
            (..) = x;


        let x = Deprecated { //~ ERROR use of unstable
            inherit: 1, //~ ERROR use of unstable
            override1: 2,
            override2: 3, //~ ERROR use of unstable
        };

        let _ = x.inherit; //~ ERROR use of unstable
        let _ = x.override1;
        let _ = x.override2; //~ ERROR use of unstable

        let Deprecated { //~ ERROR use of unstable
            inherit: _, //~ ERROR use of unstable
            override1: _,
            override2: _ //~ ERROR use of unstable
        } = x;

        let Deprecated //~ ERROR use of unstable
            // the patterns are all fine:
            { .. } = x;

        let x = Deprecated2(1, 2, 3); //~ ERROR use of unstable

        let _ = x.0; //~ ERROR use of unstable
        let _ = x.1;
        let _ = x.2; //~ ERROR use of unstable

        let Deprecated2 //~ ERROR use of unstable
            (_, //~ ERROR use of unstable
             _,
             _) //~ ERROR use of unstable
            = x;
        let Deprecated2 //~ ERROR use of unstable
            // the patterns are all fine:
            (..) = x;
    }
}

mod this_crate {
    #[stable(feature = "rust1", since = "1.0.0")]
    struct Stable {
        inherit: u8,
        #[unstable(feature = "test_feature", issue = "0")]
        override1: u8,
        #[rustc_deprecated(since = "1.0.0", reason = "text")]
        #[unstable(feature = "test_feature", issue = "0")]
        override2: u8,
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    struct Stable2(u8,
                   #[stable(feature = "rust1", since = "1.0.0")] u8,
                   #[unstable(feature = "test_feature", issue = "0")]
                   #[rustc_deprecated(since = "1.0.0", reason = "text")] u8);

    #[unstable(feature = "test_feature", issue = "0")]
    struct Unstable {
        inherit: u8,
        #[stable(feature = "rust1", since = "1.0.0")]
        override1: u8,
        #[rustc_deprecated(since = "1.0.0", reason = "text")]
        #[unstable(feature = "test_feature", issue = "0")]
        override2: u8,
    }

    #[unstable(feature = "test_feature", issue = "0")]
    struct Unstable2(u8,
                     #[stable(feature = "rust1", since = "1.0.0")] u8,
                     #[unstable(feature = "test_feature", issue = "0")]
                     #[rustc_deprecated(since = "1.0.0", reason = "text")] u8);

    #[unstable(feature = "test_feature", issue = "0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    struct Deprecated {
        inherit: u8,
        #[stable(feature = "rust1", since = "1.0.0")]
        override1: u8,
        #[unstable(feature = "test_feature", issue = "0")]
        override2: u8,
    }

    #[unstable(feature = "test_feature", issue = "0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    struct Deprecated2(u8,
                       #[stable(feature = "rust1", since = "1.0.0")] u8,
                       #[unstable(feature = "test_feature", issue = "0")] u8);

    pub fn foo() {
        let x = Stable {
            inherit: 1,
            override1: 2,
            override2: 3,
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;

        let Stable {
            inherit: _,
            override1: _,
            override2: _
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;

        let Stable2(_,
                   _,
                   _)
            = x;
        // all fine
        let Stable2(..) = x;


        let x = Unstable {
            inherit: 1,
            override1: 2,
            override2: 3,
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;

        let Unstable {
            inherit: _,
            override1: _,
            override2: _
        } = x;

        let Unstable
            // the patterns are all fine:
            { .. } = x;


        let x = Unstable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;

        let Unstable2
            (_,
             _,
             _)
            = x;
        let Unstable2
            // the patterns are all fine:
            (..) = x;


        let x = Deprecated {
            inherit: 1,
            override1: 2,
            override2: 3,
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;

        let Deprecated {
            inherit: _,
            override1: _,
            override2: _
        } = x;

        let Deprecated
            // the patterns are all fine:
            { .. } = x;

        let x = Deprecated2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;

        let Deprecated2
            (_,
             _,
             _)
            = x;
        let Deprecated2
            // the patterns are all fine:
            (..) = x;
    }
}

fn main() {}
