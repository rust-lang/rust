//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #140850
//@[next] check-pass
//@ compile-flags: -Zvalidate-mir
//@ edition:2021

// This ICE'd with the old solver as MIR build resulted in an opaque type
// as the argument to a `SwitchInt` terminator. Fixed by normalizing opaque
// types in their defining scope. Regression test for #140850.

fn foo() -> impl Sized {
    if false {
        while foo() {}
    }
    loop {}
}
fn main() {}
