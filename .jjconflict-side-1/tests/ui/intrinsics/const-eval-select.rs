//@ run-pass

#![feature(const_eval_select)]
#![feature(core_intrinsics)]

use std::intrinsics::const_eval_select;

const fn yes() -> bool {
    true
}

fn no() -> bool {
    false
}

// not allowed on stable; testing only
const fn is_const_eval() -> bool {
    const_eval_select((), yes, no)
}

fn main() {
    const YES: bool = is_const_eval();
    let no = is_const_eval();

    assert_eq!(true, YES);
    assert_eq!(false, no);
}
