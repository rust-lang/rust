// Ensure that we can successfully mangle & demangle trait object types w/ assoc const bindings.

// FIXME(mgca): Legacy mangling still crashes:
//              "finding type for [impl], encountered [crate root] with no parent"

//@ build-fail
//@ revisions: v0
//\@[legacy] compile-flags: -C symbol-mangling-version=legacy -Z unstable-options
//@    [v0] compile-flags: -C symbol-mangling-version=v0
//\@[legacy] normalize-stderr: "h[[:xdigit:]]{16}" -> "h[HASH]"
//@    [v0] normalize-stderr: "sym\[.*?\]" -> "sym[HASH]"

#![feature(min_generic_const_args, rustc_attrs)]
#![expect(incomplete_features)]
#![crate_name = "sym"]

trait Trait {
    #[type_const]
    const N: usize;
}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCs
//~| ERROR demangling(<dyn sym[
//~| ERROR demangling-alt(<dyn sym::Trait<N = 0>>)
impl dyn Trait<N = 0> {}

fn main() {}
