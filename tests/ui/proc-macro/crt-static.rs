// Test proc-macro crate can be built without additional RUSTFLAGS
// on musl target
// override -Ctarget-feature=-crt-static from compiletest
// compile-flags: --crate-type proc-macro -Ctarget-feature=
// ignore-wasm32
// ignore-sgx no support for proc-macro crate type
// build-pass
// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

// FIXME: This don't work when crate-type is specified by attribute
// `#![crate_type = "proc-macro"]`, not by `--crate-type=proc-macro`
// command line flag. This is because the list of `cfg` symbols is generated
// before attributes are parsed. See rustc_interface::util::add_configuration
#[cfg(target_feature = "crt-static")]
compile_error!("crt-static is enabled");

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Foo)]
pub fn derive_foo(input: TokenStream) -> TokenStream {
    input
}
