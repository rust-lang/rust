#![allow(while_true)]

//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    panic!("{}", {
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
