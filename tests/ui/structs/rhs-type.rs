// Tests that codegen treats the rhs of pth's decl
// as a _|_-typed thing, not a str-typed thing

//@ run-fail
//@ error-pattern:bye
//@ needs-subprocess

#![allow(unreachable_code)]
#![allow(unused_variables)]

struct T {
    t: String,
}

fn main() {
    let pth = panic!("bye");
    let _rs: T = T { t: pth };
}
