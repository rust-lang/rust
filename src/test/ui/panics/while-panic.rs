#![allow(while_true)]

// run-fail
// error-pattern:giraffe
// ignore-emscripten no processes

fn main() {
    panic!("{}", {
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
