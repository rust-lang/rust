//@ no-prefer-dynamic
//@ aux-build: decl_with_default_panics.rs
//@ edition: 2021
//@ run-pass
//@ needs-unwind
//@ exec-env:RUST_BACKTRACE=1
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// A small test to make sure that unwinding works properly.
//
// Functions can have target-cpu applied. On apple-darwin this is super important,
// since you can have binaries which mix x86 and aarch64 code that are compatible
// with both architectures. So we can't just reject target_cpu on EIIs since apple
// puts them on by default. The problem: we generate aliases. And aliases cannot
// get target_cpu applied to them. So, instead we should, in the case of functions,
// generate a shim function. For statics aliases should keep working.
// However, to make this work properly,
// on LLVM we generate shim functions instead of function aliases.
// Little extra functions that look like
// ```
// function alias_symbol(*args) {return (tailcall) aliasee(*args);}
// ```
// This is a simple test to make sure that we can unwind through these,
// and that this wrapper function effectively doesn't show up in the trace.
#![feature(extern_item_impls)]

extern crate decl_with_default_panics;

fn main() {
    let result = std::panic::catch_unwind(|| {
        decl_with_default_panics::decl1(10);
    });
    assert!(result.is_err());
}
