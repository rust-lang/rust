//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Functions can have target-cpu applied. On apple-darwin this is super important,
// since you can have binaries which mix x86 and aarch64 code that are compatible
// with both architectures. So we can't just reject target_cpu on EIIs since apple
// puts them on by default. The problem: we generate aliases. And aliases cannot
// get target_cpu applied to them. So, instead we should, in the case of functions,
// generate a shim function. For statics aliases should keep working in theory.
// In fact, aliases are only necessary for statics. For functions we could just
//  always generate a shim and a previous version of EII did so but I was sad
// that that'd never support statics.
//@ ignore-macos
// Tests EIIs with default implementations.
// When there's no explicit declaration, the default should be called from the declaring crate.
#![feature(extern_item_impls)]

extern crate decl_with_default;

fn main() {
    decl_with_default::decl1(10);
}
