// revisions: constant array
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

#[cfg(constant)]
const _: () = {
    become f(); //[constant]~ error: become statement outside of function body
};

#[cfg(array)]
struct Bad([(); become f()]); //[array]~ error: become statement outside of function body

fn f() {}

fn main() {}
