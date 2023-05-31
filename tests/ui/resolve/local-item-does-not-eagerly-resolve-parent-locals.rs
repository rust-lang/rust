// run-pass
#![allow(unused_variables, dead_code)]
const fn foo() {}

fn f1() {
    fn f() {
        foo();
    }
    const C: () = {
        foo();
    };
    static S: () = {
        foo();
    };
    let foo = 0;
}

fn f2() {
    let foo = 0;
    fn f() {
        foo();
    }
    const C: () = {
        foo();
    };
    static S: () = {
        foo();
    };
}

fn main() {}
