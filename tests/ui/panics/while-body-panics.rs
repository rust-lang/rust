#![allow(while_true)]

//@ run-fail
//@ check-run-results:quux
//@ ignore-emscripten no processes

fn main() {
    let _x: isize = {
        while true {
            panic!("quux");
        }
        8
    };
}
