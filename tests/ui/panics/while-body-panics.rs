#![allow(while_true)]

//@ run-fail
//@ error-pattern:quux
//@ needs-subprocess

fn main() {
    let _x: isize = {
        while true {
            panic!("quux");
        }
        8
    };
}
