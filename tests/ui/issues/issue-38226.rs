// run-pass
// This test makes sure that we don't run into a linker error because of the
// middle::reachable pass missing trait methods with default impls.

// aux-build:issue-38226-aux.rs

// Need -Cno-prepopulate-passes to really disable inlining, otherwise the faulty
// code gets optimized out:
// compile-flags: -Cno-prepopulate-passes -Cpasses=name-anon-globals

extern crate issue_38226_aux;

fn main() {
    issue_38226_aux::foo::<()>();
}
