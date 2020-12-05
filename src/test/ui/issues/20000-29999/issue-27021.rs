// run-pass

// This test is bogus (i.e., should be compile-fail) during the period
// where #54986 is implemented and #54987 is *not* implemented. For
// now: just ignore it
//
// ignore-test

// These are variants of issue-26996.rs. In all cases we are writing
// into a record field that has been moved out of, and ensuring that
// such a write won't overwrite the state of the thing it was moved
// into.
//
// That's a fine thing to test when this code is accepted by the
// compiler, and this code is being transcribed accordingly into
// the ui test issue-21232-partial-init-and-use.rs

fn main() {
    let mut c = (1, (1, "".to_owned()));
    match c {
        c2 => { (c.1).0 = 2; assert_eq!((c2.1).0, 1); }
    }

    let mut c = (1, (1, (1, "".to_owned())));
    match c.1 {
        c2 => { ((c.1).1).0 = 3; assert_eq!((c2.1).0, 1); }
    }
}
