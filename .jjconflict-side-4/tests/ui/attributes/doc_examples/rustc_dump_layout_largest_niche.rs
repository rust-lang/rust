//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_layout(largest_niche)]
type Alias = Option<char>;
