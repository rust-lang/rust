//@ only-elf
//@ needs-dynamic-linking

//@ only-gnu
//@ only-x86_64
//@ revisions: as_needed no_as_needed no_modifier merge_1 merge_2 merge_3 merge_4

//@ [as_needed] run-pass
//@ [no_as_needed] run-fail
//@ [no_modifier] run-pass
//@ [merge_1] run-fail
//@ [merge_2] run-fail
//@ [merge_3] run-fail
//@ [merge_4] run-pass

#![allow(incomplete_features)]
#![feature(raw_dylib_elf)]
#![feature(native_link_modifiers_as_needed)]

#[cfg_attr(
    as_needed,
    link(name = "taiqannf1y28z2rw", kind = "raw-dylib", modifiers = "+as-needed")
)]
#[cfg_attr(
    no_as_needed,
    link(name = "taiqannf1y28z2rw", kind = "raw-dylib", modifiers = "-as-needed")
)]
#[cfg_attr(no_modifier, link(name = "taiqannf1y28z2rw", kind = "raw-dylib"))]
unsafe extern "C" {}

#[cfg_attr(merge_1, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "+as-needed"))]
#[cfg_attr(merge_2, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "-as-needed"))]
#[cfg_attr(merge_3, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "-as-needed"))]
#[cfg_attr(merge_4, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "+as-needed"))]
unsafe extern "C" {}

#[cfg_attr(merge_1, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "-as-needed"))]
#[cfg_attr(merge_2, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "+as-needed"))]
#[cfg_attr(merge_3, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "-as-needed"))]
#[cfg_attr(merge_4, link(name = "k9nm7qxoa79bg7e6", kind = "raw-dylib", modifiers = "+as-needed"))]
unsafe extern "C" {}

fn main() {}
