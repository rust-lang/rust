// Regression test for issue #89396: Try to recover from a
// `=>` -> `=` or `->` typo in a match arm.

//@ run-rustfix

fn main() {
    let opt = Some(42);
    let _ = match opt {
        Some(_) = true,
        //~^ ERROR: expected one of
        //~| HELP: use a fat arrow to start a match arm
        None -> false,
        //~^ ERROR: expected one of
        //~| HELP: use a fat arrow to start a match arm
    };
}
