#![feature(rustc_attrs)]

// This test checks that a failure occurs with NLL but does not fail with the
// legacy AST output. Check issue-49824.nll.stderr for expected compilation error
// output under NLL and #49824 for more information.

#[rustc_error]
fn main() {
    //~^ compilation successful
    let mut x = 0;
    || {
        || {
            let _y = &mut x;
        }
    };
}
