// MIR optimizations can create expansions after the TyCtxt has been created.
// This test verifies that those expansions can be decoded correctly.

//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph -Z mir-opt-level=3

fn main() {
    if std::env::var("a").is_ok() {
        println!("b");
    }
}
