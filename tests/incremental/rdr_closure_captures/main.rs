// Test closures capturing private types.
// Known limitation: changes to private types captured by closures currently
// DO cause downstream rebuilds, as the closure's captured environment type
// leaks through impl Fn.
//
// - rpass1: Initial compilation
// - rpass2: Private captured type gets extra field

//@ revisions: rpass1 rpass2
//@ aux-build: closure_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

extern crate closure_dep;

fn main() {
    let closure = closure_dep::make_closure();
    assert_eq!(closure(), 42);

    let result = closure_dep::call_with_closure(|x| x * 2, 10);
    assert_eq!(result, 62);
}
