//@ add-minicore
//@ compile-flags: --target aarch64-apple-darwin
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[unsafe(link_section = "foo")]
//~^ ERROR invalid macho section specifier
#[unsafe(no_mangle)]
fn missing_section() {}

#[unsafe(link_section = "foo,")]
//~^ ERROR invalid macho section specifier
#[unsafe(no_mangle)]
fn empty_section() {}

#[unsafe(link_section = "foo, ")]
//~^ ERROR invalid macho section specifier
#[unsafe(no_mangle)]
fn whitespace_section() {}

#[unsafe(link_section = "foo,somelongwindedthing")]
//~^ ERROR invalid macho section specifier
#[unsafe(no_mangle)]
fn section_too_long() {}

#[unsafe(link_section = "foo,bar")]
#[unsafe(no_mangle)]
fn segment_and_section() {}

#[unsafe(link_section = "foo,bar,")]
#[unsafe(no_mangle)]
fn segment_and_section_and_comma() {}

#[unsafe(link_section = ",foo")]
#[unsafe(no_mangle)]
fn missing_segment_is_fine() {}

#[unsafe(link_section = "__TEXT,__stubs,symbol_stubs,none,16")]
#[unsafe(no_mangle)]
fn stub_size_decimal() {}

#[unsafe(link_section = "__TEXT,__stubs,symbol_stubs,none,0x10")]
#[unsafe(no_mangle)]
fn stub_size_hex() {}

#[unsafe(link_section = "__TEXT,__stubs,symbol_stubs,none,020")]
#[unsafe(no_mangle)]
fn stub_size_oct() {}

#[unsafe(link_section = "__TEXT,__stubs,symbol_stubs,none,020,rest,is,ignored")]
#[unsafe(no_mangle)]
fn rest_is_ignored() {}
