//@no-rustfix: the suggestion substitutes a literal into the inner fold's
// init position, which makes the inner fold lintable in a second pass, so a
// single rustfix application does not reach a fixpoint (`cargo fix` converges
// by iterating).
#![warn(clippy::unnecessary_fold)]

fn main() {
    let opt: Option<i32> = Some(2);
    let opt2: Option<i32> = Some(4);

    // Only the outer fold is linted: the inner fold's init is the outer
    // closure's accumulator parameter.
    let _ = opt.iter().fold(0, |acc, x| opt2.iter().fold(acc, |a, b| a + b) + x);
    //~^ unnecessary_fold
}
