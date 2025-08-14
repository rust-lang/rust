//@ aux-build:lint_stability_fields.rs

#![deny(deprecated)]
#![allow(dead_code)]
#![feature(staged_api, unstable_test_feature)]

#![stable(feature = "rust1", since = "1.0.0")]

mod cross_crate {
    extern crate lint_stability_fields;

    use self::lint_stability_fields::*;

    pub fn foo() {
        let x = Stable {
            inherit: 1,
            override1: 2,
            override2: 3,
            //~^ ERROR use of deprecated field
            override3: 4,
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;
        //~^ ERROR use of deprecated field
        let _ = x.override3;

        let Stable {
            inherit: _,
            override1: _,
            override2: _,
            //~^ ERROR use of deprecated field
            override3: _,
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3, 4);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;
        //~^ ERROR use of deprecated field
        let _ = x.3;

        let Stable2(_,
                   _,
                   _,
                   //~^ ERROR use of deprecated field
                   _)
            = x;
        // all fine
        let Stable2(..) = x;


        let x = Unstable {
            inherit: 1,
            override1: 2,
            override2: 3,
            //~^ ERROR use of deprecated field
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;
        //~^ ERROR use of deprecated field

        let Unstable {
            inherit: _,
            override1: _,
            override2: _
            //~^ ERROR use of deprecated field
        } = x;

        let Unstable
            // the patterns are all fine:
            { .. } = x;


        let x = Unstable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;
        //~^ ERROR use of deprecated field

        let Unstable2
            (_,
             _,
             _)
            //~^ ERROR use of deprecated field
            = x;
        let Unstable2
            // the patterns are all fine:
            (..) = x;


        let x = Deprecated {
            //~^ ERROR use of deprecated struct
            inherit: 1,
            //~^ ERROR use of deprecated field
            override1: 2,
            //~^ ERROR use of deprecated field
            override2: 3,
            //~^ ERROR use of deprecated field
        };

        let _ = x.inherit;
        //~^ ERROR use of deprecated field
        let _ = x.override1;
        //~^ ERROR use of deprecated field
        let _ = x.override2;
        //~^ ERROR use of deprecated field

        let Deprecated {
            //~^ ERROR use of deprecated struct
            inherit: _,
            //~^ ERROR use of deprecated field
            override1: _,
            //~^ ERROR use of deprecated field
            override2: _
            //~^ ERROR use of deprecated field
        } = x;

        let Deprecated
            //~^ ERROR use of deprecated struct
            // the patterns are all fine:
            { .. } = x;

        let x = Deprecated2(1, 2, 3);
        //~^ ERROR use of deprecated tuple struct

        let _ = x.0;
        //~^ ERROR use of deprecated field
        let _ = x.1;
        //~^ ERROR use of deprecated field
        let _ = x.2;
        //~^ ERROR use of deprecated field

        let Deprecated2
        //~^ ERROR use of deprecated tuple struct
            (_,
             //~^ ERROR use of deprecated field
             _,
             //~^ ERROR use of deprecated field
             _)
             //~^ ERROR use of deprecated field
            = x;
        let Deprecated2
        //~^ ERROR use of deprecated tuple struct
            // the patterns are all fine:
            (..) = x;
    }
}

mod this_crate {
    #[stable(feature = "rust1", since = "1.0.0")]
    struct Stable {
        inherit: u8,
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override1: u8,
        #[deprecated(since = "1.0.0", note = "text")]
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override2: u8,
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    struct Stable2(u8,
                   #[stable(feature = "rust1", since = "1.0.0")] u8,
                   #[unstable(feature = "unstable_test_feature", issue = "none")]
                   #[deprecated(since = "1.0.0", note = "text")] u8);

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    struct Unstable {
        inherit: u8,
        #[stable(feature = "rust1", since = "1.0.0")]
        override1: u8,
        #[deprecated(since = "1.0.0", note = "text")]
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override2: u8,
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    struct Unstable2(u8,
                     #[stable(feature = "rust1", since = "1.0.0")] u8,
                     #[unstable(feature = "unstable_test_feature", issue = "none")]
                     #[deprecated(since = "1.0.0", note = "text")] u8);

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    struct Deprecated {
        inherit: u8,
        #[stable(feature = "rust1", since = "1.0.0")]
        override1: u8,
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override2: u8,
    }

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    struct Deprecated2(u8,
                       #[stable(feature = "rust1", since = "1.0.0")] u8,
                       #[unstable(feature = "unstable_test_feature", issue = "none")] u8);

    pub fn foo() {
        let x = Stable {
            inherit: 1,
            override1: 2,
            override2: 3,
            //~^ ERROR use of deprecated field
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;
        //~^ ERROR use of deprecated field

        let Stable {
            inherit: _,
            override1: _,
            override2: _
            //~^ ERROR use of deprecated field
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;
        //~^ ERROR use of deprecated field

        let Stable2(_,
                   _,
                   _)
            //~^ ERROR use of deprecated field
            = x;
        // all fine
        let Stable2(..) = x;


        let x = Unstable {
            inherit: 1,
            override1: 2,
            override2: 3,
            //~^ ERROR use of deprecated field
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;
        //~^ ERROR use of deprecated field

        let Unstable {
            inherit: _,
            override1: _,
            override2: _
            //~^ ERROR use of deprecated field
        } = x;

        let Unstable
            // the patterns are all fine:
            { .. } = x;


        let x = Unstable2(1, 2, 3);

        let _ = x.0;
        let _ = x.1;
        let _ = x.2;
        //~^ ERROR use of deprecated field

        let Unstable2
            (_,
             _,
             _)
            //~^ ERROR use of deprecated field
            = x;
        let Unstable2
            // the patterns are all fine:
            (..) = x;


        let x = Deprecated {
            //~^ ERROR use of deprecated struct
            inherit: 1,
            //~^ ERROR use of deprecated field
            override1: 2,
            //~^ ERROR use of deprecated field
            override2: 3,
            //~^ ERROR use of deprecated field
        };

        let _ = x.inherit;
        //~^ ERROR use of deprecated field
        let _ = x.override1;
        //~^ ERROR use of deprecated field
        let _ = x.override2;
        //~^ ERROR use of deprecated field

        let Deprecated {
            //~^ ERROR use of deprecated struct
            inherit: _,
            //~^ ERROR use of deprecated field
            override1: _,
            //~^ ERROR use of deprecated field
            override2: _
            //~^ ERROR use of deprecated field
        } = x;

        let Deprecated
            //~^ ERROR use of deprecated struct
            // the patterns are all fine:
            { .. } = x;

        let x = Deprecated2(1, 2, 3);
        //~^ ERROR use of deprecated tuple struct

        let _ = x.0;
        //~^ ERROR use of deprecated field
        let _ = x.1;
        //~^ ERROR use of deprecated field
        let _ = x.2;
        //~^ ERROR use of deprecated field

        let Deprecated2
        //~^ ERROR use of deprecated tuple struct
            (_,
             //~^ ERROR use of deprecated field
             _,
             //~^ ERROR use of deprecated field
             _)
            //~^ ERROR use of deprecated field
            = x;
        let Deprecated2
        //~^ ERROR use of deprecated tuple struct
            // the patterns are all fine:
            (..) = x;
    }
}

fn main() {}
