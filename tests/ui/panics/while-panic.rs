#![allow(while_true)]

// run-fail
//@error-in-other-file:giraffe
//@ignore-target-emscripten no processes

fn main() {
    panic!("{}", {
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
