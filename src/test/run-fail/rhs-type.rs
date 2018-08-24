// Tests that codegen treats the rhs of pth's decl
// as a _|_-typed thing, not a str-typed thing
// error-pattern:bye

#![allow(unreachable_code)]
#![allow(unused_variables)]

struct T {
    t: String,
}

fn main() {
    let pth = panic!("bye");
    let _rs: T = T { t: pth };
}
