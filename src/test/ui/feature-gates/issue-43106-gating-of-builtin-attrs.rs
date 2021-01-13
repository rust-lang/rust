//~ NOTE not a function
//~^ NOTE not a foreign function or static
//~^^ NOTE not a function or static
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

// check-pass
// ignore-tidy-linelength

#![feature(test, plugin_registrar)]
#![warn(unused_attributes, unknown_lints)]
//~^ NOTE the lint level is defined here
//~| NOTE the lint level is defined here

// Exception, a gated and deprecated attribute.

#![plugin_registrar]
//~^ WARN unused attribute
//~| WARN use of deprecated attribute
//~| HELP may be removed in a future compiler version

// UNGATED WHITE-LISTED BUILT-IN ATTRIBUTES

#![warn(x5400)] //~ WARN unknown lint: `x5400`
#![allow(x5300)] //~ WARN unknown lint: `x5300`
#![forbid(x5200)] //~ WARN unknown lint: `x5200`
#![deny(x5100)] //~ WARN unknown lint: `x5100`
#![macro_use] // (allowed if no argument; see issue-43160-gating-of-macro_use.rs)
// skipping testing of cfg
// skipping testing of cfg_attr
#![should_panic] //~ WARN unused attribute
#![ignore] //~ WARN unused attribute
#![no_implicit_prelude]
#![reexport_test_harness_main = "2900"]
// see gated-link-args.rs
// see issue-43106-gating-of-macro_escape.rs for crate-level; but non crate-level is below at "2700"
// (cannot easily test gating of crate-level #[no_std]; but non crate-level is below at "2600")
#![proc_macro_derive()] //~ WARN unused attribute
#![doc = "2400"]
#![cold] //~ WARN attribute should be applied to a function
//~^ WARN
// see issue-43106-gating-of-builtin-attrs-error.rs
#![link()]
#![link_name = "1900"]
//~^ WARN attribute should be applied to a foreign function
//~^^ WARN this was previously accepted by the compiler
#![link_section = "1800"]
//~^ WARN attribute should be applied to a function or static
//~^^ WARN this was previously accepted by the compiler
// see issue-43106-gating-of-rustc_deprecated.rs
#![must_use]
// see issue-43106-gating-of-stable.rs
// see issue-43106-gating-of-unstable.rs
// see issue-43106-gating-of-deprecated.rs
#![windows_subsystem = "windows"]

// UNGATED CRATE-LEVEL BUILT-IN ATTRIBUTES

#![crate_name = "0900"]
#![crate_type = "bin"] // cannot pass "0800" here

#![crate_id = "10"]
//~^ WARN use of deprecated attribute
//~| HELP remove this attribute

// FIXME(#44232) we should warn that this isn't used.
#![feature(rust1)]
//~^ WARN no longer requires an attribute to enable
//~| NOTE `#[warn(stable_features)]` on by default

#![no_start]
//~^ WARN use of deprecated attribute
//~| HELP remove this attribute

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
    //~^ WARN unused attribute

    #[macro_use] struct S;
    //~^ WARN unused attribute

    #[macro_use] type T = S;
    //~^ WARN unused attribute

    #[macro_use] impl S { }
    //~^ WARN unused attribute
}

#[macro_export]
//~^ WARN unused attribute
mod macro_export {
    mod inner { #![macro_export] }
    //~^ WARN unused attribute

    #[macro_export] fn f() { }
    //~^ WARN unused attribute

    #[macro_export] struct S;
    //~^ WARN unused attribute

    #[macro_export] type T = S;
    //~^ WARN unused attribute

    #[macro_export] impl S { }
    //~^ WARN unused attribute
}

#[plugin_registrar]
//~^ WARN unused attribute
//~| WARN use of deprecated attribute
//~| HELP may be removed in a future compiler version
mod plugin_registrar {
    mod inner { #![plugin_registrar] }
    //~^ WARN unused attribute
    //~| WARN use of deprecated attribute
    //~| HELP may be removed in a future compiler version
    //~| NOTE `#[warn(deprecated)]` on by default

    // for `fn f()` case, see gated-plugin_registrar.rs

    #[plugin_registrar] struct S;
    //~^ WARN unused attribute
    //~| WARN use of deprecated attribute
    //~| HELP may be removed in a future compiler version

    #[plugin_registrar] type T = S;
    //~^ WARN unused attribute
    //~| WARN use of deprecated attribute
    //~| HELP may be removed in a future compiler version

    #[plugin_registrar] impl S { }
    //~^ WARN unused attribute
    //~| WARN use of deprecated attribute
    //~| HELP may be removed in a future compiler version
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

#[repr()]
mod repr {
    mod inner { #![repr()] }

    #[repr()] fn f() { }

    struct S;

    #[repr()] type T = S;

    #[repr()] impl S { }
}

#[path = "3800"]
mod path {
    mod inner { #![path="3800"] }

    #[path = "3800"] fn f() { }
    //~^ WARN unused attribute

    #[path = "3800"]  struct S;
    //~^ WARN unused attribute

    #[path = "3800"] type T = S;
    //~^ WARN unused attribute

    #[path = "3800"] impl S { }
    //~^ WARN unused attribute
}

#[automatically_derived]
//~^ WARN unused attribute
mod automatically_derived {
    mod inner { #![automatically_derived] }
    //~^ WARN unused attribute

    #[automatically_derived] fn f() { }
    //~^ WARN unused attribute

    #[automatically_derived] struct S;
    //~^ WARN unused attribute

    #[automatically_derived] type T = S;
    //~^ WARN unused attribute

    #[automatically_derived] impl S { }
    //~^ WARN unused attribute
}

#[no_mangle]
//~^ WARN attribute should be applied to a function or static [unused_attributes]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
mod no_mangle {
    //~^ NOTE not a function or static
    mod inner { #![no_mangle] }
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[no_mangle] fn f() { }

    #[no_mangle] struct S;
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[no_mangle] type T = S;
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static

    #[no_mangle] impl S { }
    //~^ WARN attribute should be applied to a function or static [unused_attributes]
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| NOTE not a function or static
}

#[should_panic]
//~^ WARN unused attribute
mod should_panic {
    mod inner { #![should_panic] }
    //~^ WARN unused attribute

    #[should_panic] fn f() { }
    //~^ WARN unused attribute

    #[should_panic] struct S;
    //~^ WARN unused attribute

    #[should_panic] type T = S;
    //~^ WARN unused attribute

    #[should_panic] impl S { }
    //~^ WARN unused attribute
}

#[ignore]
//~^ WARN unused attribute
mod ignore {
    mod inner { #![ignore] }
    //~^ WARN unused attribute

    #[ignore] fn f() { }
    //~^ WARN unused attribute

    #[ignore] struct S;
    //~^ WARN unused attribute

    #[ignore] type T = S;
    //~^ WARN unused attribute

    #[ignore] impl S { }
    //~^ WARN unused attribute
}

#[no_implicit_prelude]
//~^ WARN unused attribute
mod no_implicit_prelude {
    mod inner { #![no_implicit_prelude] }
    //~^ WARN unused attribute

    #[no_implicit_prelude] fn f() { }
    //~^ WARN unused attribute

    #[no_implicit_prelude] struct S;
    //~^ WARN unused attribute

    #[no_implicit_prelude] type T = S;
    //~^ WARN unused attribute

    #[no_implicit_prelude] impl S { }
    //~^ WARN unused attribute
}

#[reexport_test_harness_main = "2900"]
//~^ WARN unused attribute
mod reexport_test_harness_main {
    mod inner { #![reexport_test_harness_main="2900"] }
    //~^ WARN unused attribute

    #[reexport_test_harness_main = "2900"] fn f() { }
    //~^ WARN unused attribute

    #[reexport_test_harness_main = "2900"] struct S;
    //~^ WARN unused attribute

    #[reexport_test_harness_main = "2900"] type T = S;
    //~^ WARN unused attribute

    #[reexport_test_harness_main = "2900"] impl S { }
    //~^ WARN unused attribute
}

// Cannot feed "2700" to `#[macro_escape]` without signaling an error.
#[macro_escape]
//~^ WARN `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
mod macro_escape {
    mod inner { #![macro_escape] }
    //~^ WARN `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
    //~| HELP try an outer attribute: `#[macro_use]`

    #[macro_escape] fn f() { }
    //~^ WARN unused attribute

    #[macro_escape] struct S;
    //~^ WARN unused attribute

    #[macro_escape] type T = S;
    //~^ WARN unused attribute

    #[macro_escape] impl S { }
    //~^ WARN unused attribute
}

#[no_std]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod no_std {
    mod inner { #![no_std] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[no_std] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
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


// Note that this is a `check-pass` test, so it
// will never invoke the linker. These are here nonetheless to point
// out that we allow them at non-crate-level (though I do not know
// whether they have the same effect here as at crate-level).

#[link()]
mod link {
    mod inner { #![link()] }

    #[link()] fn f() { }

    #[link()] struct S;

    #[link()] type T = S;

    #[link()] impl S { }
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

#[must_use]
mod must_use {
    mod inner { #![must_use] }

    #[must_use] fn f() { }

    #[must_use] struct S;

    #[must_use] type T = S;

    #[must_use] impl S { }
}

#[windows_subsystem = "windows"]
mod windows_subsystem {
    mod inner { #![windows_subsystem="windows"] }

    #[windows_subsystem = "windows"] fn f() { }

    #[windows_subsystem = "windows"] struct S;

    #[windows_subsystem = "windows"] type T = S;

    #[windows_subsystem = "windows"] impl S { }
}

// BROKEN USES OF CRATE-LEVEL BUILT-IN ATTRIBUTES

#[crate_name = "0900"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod crate_name {
    mod inner { #![crate_name="0900"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[crate_name = "0900"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_name = "0900"] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

#[crate_type = "0800"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod crate_type {
    mod inner { #![crate_type="0800"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[crate_type = "0800"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[crate_type = "0800"] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

#[feature(x0600)]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod feature {
    mod inner { #![feature(x0600)] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[feature(x0600)] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[feature(x0600)] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}


#[no_main]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod no_main_1 {
    mod inner { #![no_main] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[no_main] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

#[no_builtins]
mod no_builtins {
    mod inner { #![no_builtins] }

    #[no_builtins] fn f() { }

    #[no_builtins] struct S;

    #[no_builtins] type T = S;

    #[no_builtins] impl S { }
}

#[recursion_limit="0200"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod recursion_limit {
    mod inner { #![recursion_limit="0200"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[recursion_limit="0200"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[recursion_limit="0200"] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

#[type_length_limit="0100"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod type_length_limit {
    mod inner { #![type_length_limit="0100"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[type_length_limit="0100"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[type_length_limit="0100"] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

fn main() {}
