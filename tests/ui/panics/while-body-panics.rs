#![allow(while_true)]

//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let _x: isize = {
        while true {
            panic!("quux");
        }
        8
    };
}
