// As documented in Issue #45983, this test is evaluating the quality
// of our diagnostics on erroneous code using higher-ranked closures.
//
// However, as documented on Issue #53026, this test also became a
// prime example of our need to test the NLL migration mode
// *separately* from the existing test suites that focus solely on
// AST-borrwock and NLL.

// revisions: ast migrate nll

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[ast]compile-flags: -Z borrowck=ast
//[migrate]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows

fn give_any<F: for<'r> FnOnce(&'r ())>(f: F) {
    f(&());
}

fn main() {
    let x = None;
    give_any(|y| x = Some(y));
    //[ast]~^ ERROR borrowed data cannot be stored outside of its closure
    //[migrate]~^^ ERROR borrowed data cannot be stored outside of its closure
    //[nll]~^^^ ERROR borrowed data escapes outside of closure
    //[nll]~| ERROR cannot assign to `x`, as it is not declared as mutable
}
