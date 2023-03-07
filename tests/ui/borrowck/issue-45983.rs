// As documented in Issue #45983, this test is evaluating the quality
// of our diagnostics on erroneous code using higher-ranked closures.

fn give_any<F: for<'r> FnOnce(&'r ())>(f: F) {
    f(&());
}

fn main() {
    let mut x = None;
    give_any(|y| x = Some(y));
    //~^ ERROR borrowed data escapes outside of closure
}
