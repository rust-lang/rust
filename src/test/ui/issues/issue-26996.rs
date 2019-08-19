// run-pass

// This test is bogus (i.e., should be compile-fail) during the period
// where #54986 is implemented and #54987 is *not* implemented. For
// now: just ignore it
//
// ignore-test

// This test is checking that the write to `c.0` (which has been moved out of)
// won't overwrite the state in `c2`.
//
// That's a fine thing to test when this code is accepted by the
// compiler, and this code is being transcribed accordingly into
// the ui test issue-21232-partial-init-and-use.rs

fn main() {
    let mut c = (1, "".to_owned());
    match c {
        c2 => {
            c.0 = 2;
            assert_eq!(c2.0, 1);
        }
    }
}
