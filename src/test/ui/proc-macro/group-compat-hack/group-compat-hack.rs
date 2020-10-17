// check-pass
// aux-build:group-compat-hack.rs
// compile-flags: -Z span-debug

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use] extern crate group_compat_hack;

// Tests the backwards compatibility hack added for certain macros
// When an attribute macro named `proc_macro_hack` or `wasm_bindgen`
// has an `NtIdent` named `$name`, we pass a plain `Ident` token in
// place of a `None`-delimited group. This allows us to maintain
// backwards compatibility for older versions of these crates.

mod no_version {
    include!("js-sys/src/lib.rs");
    include!("time-macros-impl/src/lib.rs");

    macro_rules! other {
        ($name:ident) => {
            #[my_macro] struct Three($name);
        }
    }

    struct Foo;
    impl_macros!(Foo);
    arrays!(Foo);
    other!(Foo);
}

mod with_version {
    include!("js-sys-0.3.17/src/lib.rs");
    include!("time-macros-impl-0.1.0/src/lib.rs");

    macro_rules! other {
        ($name:ident) => {
            #[my_macro] struct Three($name);
        }
    }

    struct Foo;
    impl_macros!(Foo);
    arrays!(Foo);
    other!(Foo);
}

mod actix_web_test {
    include!("actix-web/src/extract.rs");

    struct Foo;
    tuple_from_req!(Foo);
}

mod actix_web_version_test {
    include!("actix-web-2.0.0/src/extract.rs");

    struct Foo;
    tuple_from_req!(Foo);
}

mod actori_web_test {
    include!("actori-web/src/extract.rs");

    struct Foo;
    tuple_from_req!(Foo);
}

mod actori_web_version_test {
    include!("actori-web-2.0.0/src/extract.rs");

    struct Foo;
    tuple_from_req!(Foo);
}


fn main() {}
