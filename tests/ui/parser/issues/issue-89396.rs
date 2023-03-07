// Regression test for issue #89396: Try to recover from a
// `=>` -> `=` or `->` typo in a match arm.

// run-rustfix

fn main() {
    let opt = Some(42);
    let _ = match opt {
        Some(_) = true,
        //~^ ERROR: expected one of
        //~| HELP: try using a fat arrow here
        None -> false,
        //~^ ERROR: expected one of
        //~| HELP: try using a fat arrow here
    };
}
