//@ aux-build:lint_stability_fields.rs
#![allow(deprecated)]
#![allow(dead_code)]
#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod cross_crate {
    extern crate lint_stability_fields;

    mod reexport {
        #[stable(feature = "rust1", since = "1.0.0")]
        pub use super::lint_stability_fields::*;
    }

    use self::lint_stability_fields::*;

    pub fn foo() {
        let x = Stable {
            inherit: 1,
            override1: 2, //~ ERROR use of unstable
            override2: 3, //~ ERROR use of unstable
            override3: 4,
        };

        let _ = x.inherit;
        let _ = x.override1; //~ ERROR use of unstable
        let _ = x.override2; //~ ERROR use of unstable
        let _ = x.override3;

        let Stable {
            inherit: _,
            override1: _, //~ ERROR use of unstable
            override2: _, //~ ERROR use of unstable
            override3: _
        } = x;
        // all fine
        let Stable { .. } = x;

        let x = Stable2(1, 2, 3, 4);

        let _ = x.0;
        let _ = x.1; //~ ERROR use of unstable
        let _ = x.2; //~ ERROR use of unstable
        let _ = x.3;

        let Stable2(_,
                   _, //~ ERROR use of unstable
                   _, //~ ERROR use of unstable
                   _)
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

        // Unstable items are still unstable even when used through a stable "pub use".
        let x = reexport::Unstable2(1, 2, 3); //~ ERROR use of unstable

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
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override1: u8,
        #[deprecated(since = "1.0.0", note = "text")]
        #[unstable(feature = "unstable_test_feature", issue = "none")]
        override2: u8,
        #[stable(feature = "rust2", since = "2.0.0")]
        override3: u8,
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    struct Stable2(u8,
                   #[stable(feature = "rust2", since = "2.0.0")] u8,
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
            override3: 4,
        };

        let _ = x.inherit;
        let _ = x.override1;
        let _ = x.override2;
        let _ = x.override3;

        let Stable {
            inherit: _,
            override1: _,
            override2: _,
            override3: _
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
