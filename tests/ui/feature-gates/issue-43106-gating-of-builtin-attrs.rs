//~ NOTE not a function
//~| NOTE not a foreign function or static
//~| NOTE cannot be applied to crates
//~| NOTE not an `extern` block
// This test enumerates as many compiler-builtin ungated attributes as
// possible (that is, all the mutually compatible ones), and checks
// that we get "expected" (*) warnings for each in the various weird
// places that users might put them in the syntax.
//
// (*): The word "expected" is in quotes above because the cases where
// warnings are and are not emitted might not match a user's intuition
// nor the rustc developers' intent. I am really just trying to
// capture today's behavior in a test, not so that it become enshrined
// as the absolute behavior going forward, but rather so that we do
// not change the behavior in the future without even being *aware* of
// the change when it happens.
//
// At the time of authoring, the attributes here are listed in the
// order that they occur in `librustc_feature`.
//
// Any builtin attributes that:
//
//  - are not stable, or
//
//  - could not be included here covering the same cases as the other
//    attributes without raising an *error* from rustc (note though
//    that warnings are of course expected)
//
// have their own test case referenced by filename in an inline
// comment.
//
// The test feeds numeric inputs to each attribute that accepts them
// without error. We do this for two reasons: (1.) to exercise how
// inputs are handled by each, and (2.) to ease searching for related
// occurrences in the source text.

//@ check-pass

#![feature(test)]
#![warn(unused_attributes, unknown_lints)]
//~^ NOTE the lint level is defined here
//~| NOTE the lint level is defined here

// UNGATED WHITE-LISTED BUILT-IN ATTRIBUTES

#![warn(x5400)] //~ WARN unknown lint: `x5400`
#![allow(x5300)] //~ WARN unknown lint: `x5300`
#![forbid(x5200)] //~ WARN unknown lint: `x5200`
#![deny(x5100)] //~ WARN unknown lint: `x5100`
#![macro_use] // (allowed if no argument; see issue-43160-gating-of-macro_use.rs)
// skipping testing of cfg
// skipping testing of cfg_attr
#![should_panic] //~ WARN `#[should_panic]` only has an effect
#![ignore] //~ WARN `#[ignore]` only has an effect on functions
#![no_implicit_prelude]
#![reexport_test_harness_main = "2900"]
// see gated-link-args.rs
// see issue-43106-gating-of-macro_escape.rs for crate-level; but non crate-level is below at "2700"
// (cannot easily test gating of crate-level #[no_std]; but non crate-level is below at "2600")
#![proc_macro_derive()] //~ WARN `#[proc_macro_derive]` only has an effect
#![doc = "2400"]
#![cold] //~ WARN attribute should be applied to a function
//~^ WARN this was previously accepted
#![link()] //~ WARN attribute should be applied to an `extern` block
//~^ WARN this was previously accepted
#![link_name = "1900"]
//~^ WARN attribute should be applied to a foreign function
//~^^ WARN this was previously accepted by the compiler
#![link_section = "1800"]
//~^ WARN attribute should be applied to a function or static
//~^^ WARN this was previously accepted by the compiler
#![must_use]
//~^ WARN `#[must_use]` has no effect
// see issue-43106-gating-of-stable.rs
// see issue-43106-gating-of-unstable.rs
// see issue-43106-gating-of-deprecated.rs
#![windows_subsystem = "windows"]

// UNGATED CRATE-LEVEL BUILT-IN ATTRIBUTES

#![crate_name = "0900"]
#![crate_type = "bin"] // cannot pass "0800" here

// FIXME(#44232) we should warn that this isn't used.
#![feature(rust1)]
//~^ WARN no longer requires an attribute to enable
//~| NOTE `#[warn(stable_features)]` on by default

// (cannot easily gating state of crate-level #[no_main]; but non crate-level is below at "0400")
#![no_builtins]
#![recursion_limit = "0200"]
#![type_length_limit = "0100"]

// USES OF BUILT-IN ATTRIBUTES IN OTHER ("UNUSUAL") PLACES

#[warn(x5400)]
//~^ WARN unknown lint: `x5400`
mod warn {
    mod inner { #![warn(x5400)] }
    //~^ WARN unknown lint: `x5400`

    #[warn(x5400)] fn f() { }
    //~^ WARN unknown lint: `x5400`

    #[warn(x5400)] struct S;
    //~^ WARN unknown lint: `x5400`

    #[warn(x5400)] type T = S;
    //~^ WARN unknown lint: `x5400`

    #[warn(x5400)] impl S { }
    //~^ WARN unknown lint: `x5400`
}

#[allow(x5300)]
//~^ WARN unknown lint: `x5300`
mod allow {
    mod inner { #![allow(x5300)] }
    //~^ WARN unknown lint: `x5300`

    #[allow(x5300)] fn f() { }
    //~^ WARN unknown lint: `x5300`

    #[allow(x5300)] struct S;
    //~^ WARN unknown lint: `x5300`

    #[allow(x5300)] type T = S;
    //~^ WARN unknown lint: `x5300`

    #[allow(x5300)] impl S { }
    //~^ WARN unknown lint: `x5300`
}

#[forbid(x5200)]
//~^ WARN unknown lint: `x5200`
mod forbid {
    mod inner { #![forbid(x5200)] }
    //~^ WARN unknown lint: `x5200`

    #[forbid(x5200)] fn f() { }
    //~^ WARN unknown lint: `x5200`

    #[forbid(x5200)] struct S;
    //~^ WARN unknown lint: `x5200`

    #[forbid(x5200)] type T = S;
    //~^ WARN unknown lint: `x5200`

    #[forbid(x5200)] impl S { }
    //~^ WARN unknown lint: `x5200`
}

#[deny(x5100)]
//~^ WARN unknown lint: `x5100`
mod deny {
    mod inner { #![deny(x5100)] }
    //~^ WARN unknown lint: `x5100`

    #[deny(x5100)] fn f() { }
    //~^ WARN unknown lint: `x5100`

    #[deny(x5100)] struct S;
    //~^ WARN unknown lint: `x5100`

    #[deny(x5100)] type T = S;
    //~^ WARN unknown lint: `x5100`

    #[deny(x5100)] impl S { }
    //~^ WARN unknown lint: `x5100`
}

#[macro_use]
mod macro_use {
    mod inner { #![macro_use] }

    #[macro_use] fn f() { }
    //~^ WARN `#[macro_use]` only has an effect

    #[macro_use] struct S;
    //~^ WARN `#[macro_use]` only has an effect

    #[macro_use] type T = S;
    //~^ WARN `#[macro_use]` only has an effect

    #[macro_use] impl S { }
    //~^ WARN `#[macro_use]` only has an effect
}

#[macro_export]
//~^ WARN `#[macro_export]` only has an effect on macro definitions
mod macro_export {
    mod inner { #![macro_export] }
    //~^ WARN `#[macro_export]` only has an effect on macro definitions

    #[macro_export] fn f() { }
    //~^ WARN `#[macro_export]` only has an effect on macro definitions

    #[macro_export] struct S;
    //~^ WARN `#[macro_export]` only has an effect on macro definitions

    #[macro_export] type T = S;
    //~^ WARN `#[macro_export]` only has an effect on macro definitions

    #[macro_export] impl S { }
    //~^ WARN `#[macro_export]` only has an effect on macro definitions
}

// At time of unit test authorship, if compiling without `--test` then
// non-crate-level #[test] attributes seem to be ignored.

#[test]
mod test { mod inner { #![test] }

    fn f() { }

    struct S;

    type T = S;

    impl S { }
}

// At time of unit test authorship, if compiling without `--test` then
// non-crate-level #[bench] attributes seem to be ignored.

#[bench]
mod bench {
    mod inner { #![bench] }

    #[bench]
    struct S;

    #[bench]
    type T = S;

    #[bench]
    impl S { }
}

#[path = "3800"]
mod path {
    mod inner { #![path="3800"] }

    #[path = "3800"] fn f() { }
    //~^ WARN `#[path]` only has an effect

    #[path = "3800"]  struct S;
    //~^ WARN `#[path]` only has an effect

    #[path = "3800"] type T = S;
    //~^ WARN `#[path]` only has an effect

    #[path = "3800"] impl S { }
    //~^ WARN `#[path]` only has an effect
}

#[automatically_derived]
//~^ WARN `#[automatically_derived]` only has an effect
mod automatically_derived {
    mod inner { #![automatically_derived] }
    //~^ WARN `#[automatically_derived]

    #[automatically_derived] fn f() { }
    //~^ WARN `#[automatically_derived]

    #[automatically_derived] struct S;
    //~^ WARN `#[automatically_derived]

    #[automatically_derived] type T = S;
    //~^ WARN `#[automatically_derived]

    #[automatically_derived] impl S { }
}

#[no_mangle]
//~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
mod no_mangle {
    //~^ NOTE not a free function, impl method or static
    mod inner { #![no_mangle] }
    //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a free function, impl method or static

    #[no_mangle] fn f() { }

    #[no_mangle] struct S;
    //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a free function, impl method or static

    #[no_mangle] type T = S;
    //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a free function, impl method or static

    #[no_mangle] impl S { }
    //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a free function, impl method or static

    trait Tr {
        #[no_mangle] fn foo();
        //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
        //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        //~| NOTE not a free function, impl method or static

        #[no_mangle] fn bar() {}
        //~^ WARN attribute should be applied to a free function, impl method or static [unused_attributes]
        //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        //~| NOTE not a free function, impl method or static
    }
}

#[should_panic]
//~^ WARN `#[should_panic]` only has an effect on
mod should_panic {
    mod inner { #![should_panic] }
    //~^ WARN `#[should_panic]` only has an effect on

    #[should_panic] fn f() { }

    #[should_panic] struct S;
    //~^ WARN `#[should_panic]` only has an effect on

    #[should_panic] type T = S;
    //~^ WARN `#[should_panic]` only has an effect on

    #[should_panic] impl S { }
    //~^ WARN `#[should_panic]` only has an effect on
}

#[ignore]
//~^ WARN `#[ignore]` only has an effect on functions
mod ignore {
    mod inner { #![ignore] }
    //~^ WARN `#[ignore]` only has an effect on functions

    #[ignore] fn f() { }

    #[ignore] struct S;
    //~^ WARN `#[ignore]` only has an effect on functions

    #[ignore] type T = S;
    //~^ WARN `#[ignore]` only has an effect on functions

    #[ignore] impl S { }
    //~^ WARN `#[ignore]` only has an effect on functions
}

#[no_implicit_prelude]
mod no_implicit_prelude {
    mod inner { #![no_implicit_prelude] }

    #[no_implicit_prelude] fn f() { }
    //~^ WARN `#[no_implicit_prelude]` only has an effect

    #[no_implicit_prelude] struct S;
    //~^ WARN `#[no_implicit_prelude]` only has an effect

    #[no_implicit_prelude] type T = S;
    //~^ WARN `#[no_implicit_prelude]` only has an effect

    #[no_implicit_prelude] impl S { }
    //~^ WARN `#[no_implicit_prelude]` only has an effect
}

#[reexport_test_harness_main = "2900"]
//~^ WARN crate-level attribute should be
mod reexport_test_harness_main {
    mod inner { #![reexport_test_harness_main="2900"] }
    //~^ WARN crate-level attribute should be

    #[reexport_test_harness_main = "2900"] fn f() { }
    //~^ WARN crate-level attribute should be

    #[reexport_test_harness_main = "2900"] struct S;
    //~^ WARN crate-level attribute should be

    #[reexport_test_harness_main = "2900"] type T = S;
    //~^ WARN crate-level attribute should be

    #[reexport_test_harness_main = "2900"] impl S { }
    //~^ WARN crate-level attribute should be
}

// Cannot feed "2700" to `#[macro_escape]` without signaling an error.
#[macro_escape]
//~^ WARN `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
mod macro_escape {
    mod inner { #![macro_escape] }
    //~^ WARN `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
    //~| HELP try an outer attribute: `#[macro_use]`

    #[macro_escape] fn f() { }
    //~^ WARN `#[macro_escape]` only has an effect

    #[macro_escape] struct S;
    //~^ WARN `#[macro_escape]` only has an effect

    #[macro_escape] type T = S;
    //~^ WARN `#[macro_escape]` only has an effect

    #[macro_escape] impl S { }
    //~^ WARN `#[macro_escape]` only has an effect
}

#[no_std]
//~^ WARN crate-level attribute should be an inner attribute
mod no_std {
    mod inner { #![no_std] }
//~^ WARN crate-level attribute should be in the root module

    #[no_std] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_std] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_std] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_std] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

// At time of authorship, #[proc_macro_derive = "2500"] signals error
// when it occurs on a mod (apart from crate-level). Therefore it goes
// into its own file; see issue-43106-gating-of-proc_macro_derive.rs

#[doc = "2400"]
mod doc {
    mod inner { #![doc="2400"] }

    #[doc = "2400"] fn f() { }

    #[doc = "2400"] struct S;

    #[doc = "2400"] type T = S;

    #[doc = "2400"] impl S { }
}

#[cold]
//~^ WARN attribute should be applied to a function
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
mod cold {
    //~^ NOTE not a function

    mod inner { #![cold] }
    //~^ WARN attribute should be applied to a function
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function

    #[cold] fn f() { }

    #[cold] struct S;
    //~^ WARN attribute should be applied to a function
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function

    #[cold] type T = S;
    //~^ WARN attribute should be applied to a function
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function

    #[cold] impl S { }
    //~^ WARN attribute should be applied to a function
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function
}

#[link_name = "1900"]
//~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
mod link_name {
    //~^ NOTE not a foreign function or static

    #[link_name = "1900"]
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| HELP try `#[link(name = "1900")]` instead
    extern "C" { }
    //~^ NOTE not a foreign function or static

    mod inner { #![link_name="1900"] }
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] fn f() { }
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] struct S;
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] type T = S;
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] impl S { }
    //~^ WARN attribute should be applied to a foreign function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a foreign function or static
}

#[link_section = "1800"]
//~^ WARN attribute should be applied to a function or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
mod link_section {
    //~^ NOTE not a function or static

    mod inner { #![link_section="1800"] }
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[link_section = "1800"] fn f() { }

    #[link_section = "1800"] struct S;
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[link_section = "1800"] type T = S;
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[link_section = "1800"] impl S { }
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static
}


// Note that this is a `check-pass` test, so it will never invoke the linker.

#[link()]
//~^ WARN attribute should be applied to an `extern` block
//~| WARN this was previously accepted
mod link {
    //~^ NOTE not an `extern` block

    mod inner { #![link()] }
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
    //~| NOTE not an `extern` block

    #[link()] fn f() { }
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
    //~| NOTE not an `extern` block

    #[link()] struct S;
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
    //~| NOTE not an `extern` block

    #[link()] type T = S;
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
    //~| NOTE not an `extern` block

    #[link()] impl S { }
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
    //~| NOTE not an `extern` block

    #[link()] extern "Rust" {}
    //~^ WARN attribute should be applied to an `extern` block
    //~| WARN this was previously accepted
}

struct StructForDeprecated;

#[deprecated]
mod deprecated {
    mod inner { #![deprecated] }

    #[deprecated] fn f() { }

    #[deprecated] struct S1;

    #[deprecated] type T = super::StructForDeprecated;

    #[deprecated] impl super::StructForDeprecated { }
}

#[must_use] //~ WARN `#[must_use]` has no effect
mod must_use {
    mod inner { #![must_use] } //~ WARN `#[must_use]` has no effect

    #[must_use] fn f() { }

    #[must_use] struct S;

    #[must_use] type T = S; //~ WARN `#[must_use]` has no effect

    #[must_use] impl S { } //~ WARN `#[must_use]` has no effect
}

#[windows_subsystem = "windows"]
//~^ WARN crate-level attribute should be an inner attribute
mod windows_subsystem {
    mod inner { #![windows_subsystem="windows"] }
    //~^ WARN crate-level attribute should be in the root module

    #[windows_subsystem = "windows"] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[windows_subsystem = "windows"] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[windows_subsystem = "windows"] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[windows_subsystem = "windows"] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

// BROKEN USES OF CRATE-LEVEL BUILT-IN ATTRIBUTES

#[crate_name = "0900"]
//~^ WARN crate-level attribute should be an inner attribute
mod crate_name {
    mod inner { #![crate_name="0900"] }
//~^ WARN crate-level attribute should be in the root module

    #[crate_name = "0900"] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

#[crate_type = "0800"]
//~^ WARN crate-level attribute should be an inner attribute
mod crate_type {
    mod inner { #![crate_type="0800"] }
//~^ WARN crate-level attribute should be in the root module

    #[crate_type = "0800"] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

#[feature(x0600)]
//~^ WARN crate-level attribute should be an inner attribute
mod feature {
    mod inner { #![feature(x0600)] }
//~^ WARN crate-level attribute should be in the root module

    #[feature(x0600)] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}


#[no_main]
//~^ WARN crate-level attribute should be an inner attribute
mod no_main_1 {
    mod inner { #![no_main] }
//~^ WARN crate-level attribute should be in the root module

    #[no_main] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_main] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_main] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_main] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

#[no_builtins]
//~^ WARN crate-level attribute should be an inner attribute
mod no_builtins {
    mod inner { #![no_builtins] }
    //~^ WARN crate-level attribute should be in the root module

    #[no_builtins] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_builtins] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_builtins] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[no_builtins] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

#[recursion_limit="0200"]
//~^ WARN crate-level attribute should be an inner attribute
mod recursion_limit {
    mod inner { #![recursion_limit="0200"] }
//~^ WARN crate-level attribute should be in the root module

    #[recursion_limit="0200"] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

#[type_length_limit="0100"]
//~^ WARN crate-level attribute should be an inner attribute
mod type_length_limit {
    mod inner { #![type_length_limit="0100"] }
//~^ WARN crate-level attribute should be in the root module

    #[type_length_limit="0100"] fn f() { }
    //~^ WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] struct S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] type T = S;
    //~^ WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] impl S { }
    //~^ WARN crate-level attribute should be an inner attribute
}

fn main() {}
