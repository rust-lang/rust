// Verifies that invalid user-defined allocation token hints (i.e., contains-pointer classification
// and type name encoding) can not be used.
//
//@ needs-sanitizer-alloc-token
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=alloc-token -Cunsafe-allow-abi-mismatch=sanitizer

#![feature(alloc_token_hint, no_core)]
#![no_core]
#![no_main]

#[alloc_token_hint] //~ ERROR malformed `alloc_token_hint` attribute input
pub struct Type1(i32);

#[alloc_token_hint()] //~ ERROR requires at least one of `contains_pointers` or `type_name`
pub struct Type2(i32);

#[alloc_token_hint(contains_pointers = "false")] //~ ERROR expected a boolean literal
pub struct Type3(i32);

#[alloc_token_hint(foo = true)] //~ ERROR malformed `alloc_token_hint` attribute input
pub struct Type4(i32);

#[alloc_token_hint(contains_pointers = true, contains_pointers = false)] //~ ERROR malformed `alloc_token_hint` attribute input
pub struct Type5(i32);
