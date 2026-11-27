//@ check-pass
//@ compile-flags: -W unreachable-code
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn callee() -> ! {
    loop {}
}

fn caller() -> ! {
    become callee();
}

fn caller_stmt() -> ! {
    become callee();
    let _x = 2580;
    //~^ WARN unreachable statement
}

fn caller_become() -> ! {
    become callee();
    become callee();
    //~^ WARN unreachable statement
}

fn main() {}
