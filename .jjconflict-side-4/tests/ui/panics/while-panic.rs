#![allow(while_true)]

//@ run-fail
//@ error-pattern:giraffe
//@ needs-subprocess

fn main() {
    panic!("{}", {
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
