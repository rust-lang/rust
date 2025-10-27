//@ check-pass
//@ rustc-env:CARGO_CRATE_NAME=build_script_build
//@ compile-flags:-Cremark=all -Cdebuginfo=1
//@ compile-flags:--crate-name=build_script_build

fn main() {
    _ = "".split('.');
}
