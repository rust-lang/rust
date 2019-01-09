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
// order that they occur in libsyntax/feature_gate.rs.
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

// skip-codegen
#![warn(unused_attributes, unknown_lints)]
#![allow(dead_code)]
#![allow(stable_features)]

// UNGATED WHITE-LISTED BUILT-IN ATTRIBUTES

#![warn                        (x5400)] //~ WARN unknown lint: `x5400`
#![allow                       (x5300)] //~ WARN unknown lint: `x5300`
#![forbid                      (x5200)] //~ WARN unknown lint: `x5200`
#![deny                        (x5100)] //~ WARN unknown lint: `x5100`
#![macro_use] // (allowed if no argument; see issue-43160-gating-of-macro_use.rs)
#![macro_export               = "4800"] //~ WARN unused attribute
#![plugin_registrar           = "4700"] //~ WARN unused attribute
// skipping testing of cfg
// skipping testing of cfg_attr
#![main                      = "x4400"] //~ WARN unused attribute
#![start                     = "x4300"] //~ WARN unused attribute
// see issue-43106-gating-of-test.rs for crate-level; but non crate-level is below at "4200"
// see issue-43106-gating-of-bench.rs for crate-level; but non crate-level is below at "4100"
#![repr                       = "3900"]
//~^ WARN unused attribute
//~| WARN `repr` attribute isn't configurable with a literal
#![path                       = "3800"] //~ WARN unused attribute
#![abi                        = "3700"] //~ WARN unused attribute
#![automatically_derived      = "3600"] //~ WARN unused attribute
#![no_mangle                  = "3500"]
#![no_link                    = "3400"] //~ WARN unused attribute
// see issue-43106-gating-of-derive.rs
#![should_panic               = "3200"] //~ WARN unused attribute
#![ignore                     = "3100"] //~ WARN unused attribute
#![no_implicit_prelude        = "3000"]
#![reexport_test_harness_main = "2900"]
// see gated-link-args.rs
// see issue-43106-gating-of-macro_escape.rs for crate-level; but non crate-level is below at "2700"
// (cannot easily test gating of crate-level #[no_std]; but non crate-level is below at "2600")
#![proc_macro_derive          = "2500"] //~ WARN unused attribute
#![doc                        = "2400"]
#![cold                       = "2300"]
#![export_name                = "2200"]
// see issue-43106-gating-of-inline.rs
#![link                       = "2000"]
#![link_name                  = "1900"]
#![link_section               = "1800"]
#![no_builtins                = "1700"] // Yikes, dupe'd on BUILTIN_ATTRIBUTES list (see "0300")
#![no_mangle                  = "1600"] // Yikes, dupe'd on BUILTIN_ATTRIBUTES list (see "3500")
// see issue-43106-gating-of-rustc_deprecated.rs
#![must_use                   = "1400"]
// see issue-43106-gating-of-stable.rs
// see issue-43106-gating-of-unstable.rs
// see issue-43106-gating-of-deprecated.rs
#![windows_subsystem          = "1000"]

// UNGATED CRATE-LEVEL BUILT-IN ATTRIBUTES

#![crate_name                 = "0900"]
#![crate_type                 = "bin"] // cannot pass "0800" here

// For #![crate_id], see issue #43142. (I cannot bear to enshrine current behavior in a test)

// FIXME(#44232) we should warn that this isn't used.
#![feature                    ( rust1)]

// For #![no_start], see issue #43144. (I cannot bear to enshrine current behavior in a test)

// (cannot easily gating state of crate-level #[no_main]; but non crate-level is below at "0400")
#![no_builtins                = "0300"]
#![recursion_limit            = "0200"]
#![type_length_limit          = "0100"]

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

#[macro_export = "4800"]
//~^ WARN unused attribute
mod macro_export {
    mod inner { #![macro_export="4800"] }
    //~^ WARN unused attribute

    #[macro_export = "4800"] fn f() { }
    //~^ WARN unused attribute

    #[macro_export = "4800"] struct S;
    //~^ WARN unused attribute

    #[macro_export = "4800"] type T = S;
    //~^ WARN unused attribute

    #[macro_export = "4800"] impl S { }
    //~^ WARN unused attribute
}

#[plugin_registrar = "4700"]
//~^ WARN unused attribute
mod plugin_registrar {
    mod inner { #![plugin_registrar="4700"] }
    //~^ WARN unused attribute

    // for `fn f()` case, see gated-plugin_registrar.rs

    #[plugin_registrar = "4700"] struct S;
    //~^ WARN unused attribute

    #[plugin_registrar = "4700"] type T = S;
    //~^ WARN unused attribute

    #[plugin_registrar = "4700"] impl S { }
    //~^ WARN unused attribute
}

#[main = "4400"]
//~^ WARN unused attribute
mod main {
    mod inner { #![main="4300"] }
    //~^ WARN unused attribute

    // for `fn f()` case, see feature-gate-main.rs

    #[main = "4400"] struct S;
    //~^ WARN unused attribute

    #[main = "4400"] type T = S;
    //~^ WARN unused attribute

    #[main = "4400"] impl S { }
    //~^ WARN unused attribute
}

#[start = "4300"]
//~^ WARN unused attribute
mod start {
    mod inner { #![start="4300"] }
    //~^ WARN unused attribute

    // for `fn f()` case, see feature-gate-start.rs

    #[start = "4300"] struct S;
    //~^ WARN unused attribute

    #[start = "4300"] type T = S;
    //~^ WARN unused attribute

    #[start = "4300"] impl S { }
    //~^ WARN unused attribute
}

// At time of unit test authorship, if compiling without `--test` then
// non-crate-level #[test] attributes seem to be ignored.

#[test = "4200"]
mod test { mod inner { #![test="4200"] }

    fn f() { }

    struct S;

    type T = S;

    impl S { }
}

// At time of unit test authorship, if compiling without `--test` then
// non-crate-level #[bench] attributes seem to be ignored.

#[bench = "4100"]
mod bench {
    mod inner { #![bench="4100"] }

    #[bench = "4100"]
    struct S;

    #[bench = "4100"]
    type T = S;

    #[bench = "4100"]
    impl S { }
}

#[repr = "3900"]
//~^ WARN unused attribute
//~| WARN `repr` attribute isn't configurable with a literal
mod repr {
    mod inner { #![repr="3900"] }
    //~^ WARN unused attribute
    //~| WARN `repr` attribute isn't configurable with a literal

    #[repr = "3900"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN `repr` attribute isn't configurable with a literal

    struct S;

    #[repr = "3900"] type T = S;
    //~^ WARN unused attribute
    //~| WARN `repr` attribute isn't configurable with a literal

    #[repr = "3900"] impl S { }
    //~^ WARN unused attribute
    //~| WARN `repr` attribute isn't configurable with a literal
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

#[abi = "3700"]
//~^ WARN unused attribute
mod abi {
    mod inner { #![abi="3700"] }
    //~^ WARN unused attribute

    #[abi = "3700"] fn f() { }
    //~^ WARN unused attribute

    #[abi = "3700"] struct S;
    //~^ WARN unused attribute

    #[abi = "3700"] type T = S;
    //~^ WARN unused attribute

    #[abi = "3700"] impl S { }
    //~^ WARN unused attribute
}

#[automatically_derived = "3600"]
//~^ WARN unused attribute
mod automatically_derived {
    mod inner { #![automatically_derived="3600"] }
    //~^ WARN unused attribute

    #[automatically_derived = "3600"] fn f() { }
    //~^ WARN unused attribute

    #[automatically_derived = "3600"] struct S;
    //~^ WARN unused attribute

    #[automatically_derived = "3600"] type T = S;
    //~^ WARN unused attribute

    #[automatically_derived = "3600"] impl S { }
    //~^ WARN unused attribute
}

#[no_mangle = "3500"]
mod no_mangle {
    mod inner { #![no_mangle="3500"] }

    #[no_mangle = "3500"] fn f() { }

    #[no_mangle = "3500"] struct S;

    #[no_mangle = "3500"] type T = S;

    #[no_mangle = "3500"] impl S { }
}

#[no_link = "3400"]
//~^ WARN unused attribute
mod no_link {
    mod inner { #![no_link="3400"] }
    //~^ WARN unused attribute

    #[no_link = "3400"] fn f() { }
    //~^ WARN unused attribute

    #[no_link = "3400"] struct S;
    //~^ WARN unused attribute

    #[no_link = "3400"]type T = S;
    //~^ WARN unused attribute

    #[no_link = "3400"] impl S { }
    //~^ WARN unused attribute
}

#[should_panic = "3200"]
//~^ WARN unused attribute
mod should_panic {
    mod inner { #![should_panic="3200"] }
    //~^ WARN unused attribute

    #[should_panic = "3200"] fn f() { }
    //~^ WARN unused attribute

    #[should_panic = "3200"] struct S;
    //~^ WARN unused attribute

    #[should_panic = "3200"] type T = S;
    //~^ WARN unused attribute

    #[should_panic = "3200"] impl S { }
    //~^ WARN unused attribute
}

#[ignore = "3100"]
//~^ WARN unused attribute
mod ignore {
    mod inner { #![ignore="3100"] }
    //~^ WARN unused attribute

    #[ignore = "3100"] fn f() { }
    //~^ WARN unused attribute

    #[ignore = "3100"] struct S;
    //~^ WARN unused attribute

    #[ignore = "3100"] type T = S;
    //~^ WARN unused attribute

    #[ignore = "3100"] impl S { }
    //~^ WARN unused attribute
}

#[no_implicit_prelude = "3000"]
//~^ WARN unused attribute
mod no_implicit_prelude {
    mod inner { #![no_implicit_prelude="3000"] }
    //~^ WARN unused attribute

    #[no_implicit_prelude = "3000"] fn f() { }
    //~^ WARN unused attribute

    #[no_implicit_prelude = "3000"] struct S;
    //~^ WARN unused attribute

    #[no_implicit_prelude = "3000"] type T = S;
    //~^ WARN unused attribute

    #[no_implicit_prelude = "3000"] impl S { }
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
//~^ WARN macro_escape is a deprecated synonym for macro_use
mod macro_escape {
    mod inner { #![macro_escape] }
    //~^ WARN macro_escape is a deprecated synonym for macro_use

    #[macro_escape] fn f() { }
    //~^ WARN unused attribute

    #[macro_escape] struct S;
    //~^ WARN unused attribute

    #[macro_escape] type T = S;
    //~^ WARN unused attribute

    #[macro_escape] impl S { }
    //~^ WARN unused attribute
}

#[no_std = "2600"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod no_std {
    mod inner { #![no_std="2600"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[no_std = "2600"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std = "2600"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std = "2600"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_std = "2600"] impl S { }
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

#[cold = "2300"]
mod cold {
    mod inner { #![cold="2300"] }

    #[cold = "2300"] fn f() { }

    #[cold = "2300"] struct S;

    #[cold = "2300"] type T = S;

    #[cold = "2300"] impl S { }
}

#[export_name = "2200"]
mod export_name {
    mod inner { #![export_name="2200"] }

    #[export_name = "2200"] fn f() { }

    #[export_name = "2200"] struct S;

    #[export_name = "2200"] type T = S;

    #[export_name = "2200"] impl S { }
}

// Note that this test has a `skip-codegen`, so it
// will never invoke the linker. These are here nonetheless to point
// out that we allow them at non-crate-level (though I do not know
// whether they have the same effect here as at crate-level).

#[link = "2000"]
mod link {
    mod inner { #![link="2000"] }

    #[link = "2000"] fn f() { }

    #[link = "2000"] struct S;

    #[link = "2000"] type T = S;

    #[link = "2000"] impl S { }
}

#[link_name = "1900"]
mod link_name {
    mod inner { #![link_name="1900"] }

    #[link_name = "1900"] fn f() { }

    #[link_name = "1900"] struct S;

    #[link_name = "1900"] type T = S;

    #[link_name = "1900"] impl S { }
}

#[link_section = "1800"]
mod link_section {
    mod inner { #![link_section="1800"] }

    #[link_section = "1800"] fn f() { }

    #[link_section = "1800"] struct S;

    #[link_section = "1800"] type T = S;

    #[link_section = "1800"] impl S { }
}

struct StructForDeprecated;

#[deprecated = "1500"]
mod deprecated {
    mod inner { #![deprecated="1500"] }

    #[deprecated = "1500"] fn f() { }

    #[deprecated = "1500"] struct S1;

    #[deprecated = "1500"] type T = super::StructForDeprecated;

    #[deprecated = "1500"] impl super::StructForDeprecated { }
}

#[must_use = "1400"]
mod must_use {
    mod inner { #![must_use="1400"] }

    #[must_use = "1400"] fn f() { }

    #[must_use = "1400"] struct S;

    #[must_use = "1400"] type T = S;

    #[must_use = "1400"] impl S { }
}

#[windows_subsystem = "1000"]
mod windows_subsystem {
    mod inner { #![windows_subsystem="1000"] }

    #[windows_subsystem = "1000"] fn f() { }

    #[windows_subsystem = "1000"] struct S;

    #[windows_subsystem = "1000"] type T = S;

    #[windows_subsystem = "1000"] impl S { }
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


#[no_main = "0400"]
//~^ WARN unused attribute
//~| WARN crate-level attribute should be an inner attribute
mod no_main_1 {
    mod inner { #![no_main="0400"] }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be in the root module

    #[no_main = "0400"] fn f() { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main = "0400"] struct S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main = "0400"] type T = S;
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute

    #[no_main = "0400"] impl S { }
    //~^ WARN unused attribute
    //~| WARN crate-level attribute should be an inner attribute
}

#[no_builtins = "0300"]
mod no_builtins {
    mod inner { #![no_builtins="0200"] }

    #[no_builtins = "0300"] fn f() { }

    #[no_builtins = "0300"] struct S;

    #[no_builtins = "0300"] type T = S;

    #[no_builtins = "0300"] impl S { }
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







fn main() {
    println!("Hello World");
}
