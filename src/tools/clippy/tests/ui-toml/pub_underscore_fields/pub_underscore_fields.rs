//@revisions: exported all_pub_fields
//@[all_pub_fields] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/pub_underscore_fields/all_pub_fields
//@[exported] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/pub_underscore_fields/exported

#![allow(unused)]
#![warn(clippy::pub_underscore_fields)]

use std::marker::PhantomData;

pub mod inner {
    use std::marker;

    pub struct PubSuper {
        pub(super) a: usize,
        pub _b: u8,
        //~^ pub_underscore_fields
        _c: i32,
        pub _mark: marker::PhantomData<u8>,
    }

    mod inner2 {
        pub struct PubModVisibility {
            pub(in crate::inner) e: bool,
            pub(in crate::inner) _f: Option<()>,
            //~[all_pub_fields]^ pub_underscore_fields
        }

        struct PrivateStructPubField {
            pub _g: String,
            //~[all_pub_fields]^ pub_underscore_fields
        }
    }
}

fn main() {
    pub struct StructWithOneViolation {
        pub _a: usize,
        //~[all_pub_fields]^ pub_underscore_fields
    }

    // should handle structs with multiple violations
    pub struct StructWithMultipleViolations {
        a: u8,
        _b: usize,
        pub _c: i64,
        //~[all_pub_fields]^ pub_underscore_fields
        #[doc(hidden)]
        pub d: String,
        pub _e: Option<u8>,
        //~[all_pub_fields]^ pub_underscore_fields
    }

    // shouldn't warn on anonymous fields
    pub struct AnonymousFields(pub usize, i32);

    // don't warn on empty structs
    pub struct Empty1;
    pub struct Empty2();
    pub struct Empty3 {};

    pub struct PubCrate {
        pub(crate) a: String,
        pub(crate) _b: Option<String>,
        //~[all_pub_fields]^ pub_underscore_fields
    }

    // shouldn't warn on fields named pub
    pub struct NamedPub {
        r#pub: bool,
        _pub: String,
        pub(crate) _mark: PhantomData<u8>,
    }

    // shouldn't warn when `#[allow]` is used on field level
    pub struct AllowedViolations {
        #[allow(clippy::pub_underscore_fields)]
        pub _first: u32,
    }
}
