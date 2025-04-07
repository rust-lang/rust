//@ revisions: feature_disabled feature_enabled
#![cfg_attr(feature_enabled, feature(unboxed_closures))]

// rust-call is unstable and not enabled, so it should not be suggested as a fix
extern "rust-cull" fn rust_call(_: ()) {}
//~^ ERROR invalid ABI
//[feature_enabled]~| HELP there's a similarly named valid ABI

fn main() {
    rust_call(());
}
