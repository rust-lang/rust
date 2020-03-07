// ignore-tidy-linelength
// only-x86_64
// check-pass
// NOTE: We always compile this test with -Copt-level=0 because higher opt-levels
//       optimize away the const function
// compile-flags:-Copt-level=0
#![feature(const_eval_limit)]
#![const_eval_limit="2"]

const CONSTANT: usize = limit();
//~^ WARNING Constant evaluating a complex constant, this might take some time

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize { //~ WARNING Constant evaluating a complex constant, this might take some time
    let x = 42;

    x * 42
}
