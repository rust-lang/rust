// run-pass
// Ensure that declarations and types which use `extern fn` both have the same
// ABI (#9309).

// pretty-expanded FIXME #23616
// aux-build:fn-abi.rs

extern crate fn_abi;

extern {
    fn foo();
}

pub fn main() {
    // Will only type check if the type of _p and the decl of foo use the
    // same ABI
    let _p: unsafe extern fn() = foo;
}
