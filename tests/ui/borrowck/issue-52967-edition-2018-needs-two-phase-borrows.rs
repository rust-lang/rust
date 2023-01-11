// This is a regression test for #52967, where we discovered that in
// the initial deployment of NLL for the 2018 edition, I forgot to
// turn on two-phase-borrows in addition to `-Z borrowck=migrate`.

// revisions: edition2015 edition2018
//[edition2018]edition:2018

// run-pass

fn the_bug() {
    let mut stuff = ("left", "right");
    match stuff {
        (ref mut left, _) if *left == "left" => { *left = "new left"; }
        _ => {}
    }
    assert_eq!(stuff, ("new left", "right"));
}

fn main() {
    the_bug();
}
