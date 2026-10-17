//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
//@ normalize-stderr: "DefId\((\d+):(\d+)" -> "DefId(..:.."
//@ normalize-stderr: "\[[A-Fa-f0-9]{4}\]" -> "[....]"

#![feature(rustc_attrs)]

#[rustc_dump_generics]
struct Struct<'lifetime, const CONST: usize, GENERIC> {
    stuff: &'lifetime [GENERIC; CONST],
}
