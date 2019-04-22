// As documented in Issue #45983, this test is evaluating the quality
// of our diagnostics on erroneous code using higher-ranked closures.

// revisions: migrate nll

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[nll]compile-flags: -Z borrowck=mir

fn give_any<F: for<'r> FnOnce(&'r ())>(f: F) {
    f(&());
}

fn main() {
    let x = None;
    give_any(|y| x = Some(y));
    //[migrate]~^ ERROR borrowed data cannot be stored outside of its closure
    //[nll]~^^ ERROR borrowed data escapes outside of closure
    //[nll]~| ERROR cannot assign to `x`, as it is not declared as mutable
}
