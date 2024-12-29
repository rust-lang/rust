#![allow(while_true)]

//@ run-fail
//@ check-run-results:giraffe
//@ ignore-emscripten no processes

fn main() {
    panic!("{}", {
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
