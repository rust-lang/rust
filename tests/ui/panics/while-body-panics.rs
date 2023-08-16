#![allow(while_true)]

// run-fail
//@error-in-other-file:quux
//@ignore-target-emscripten no processes

fn main() {
    let _x: isize = {
        while true {
            panic!("quux");
        }
        8
    };
}
