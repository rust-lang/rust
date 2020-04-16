#![allow(while_true)]

// run-fail
// error-pattern:giraffe

fn main() {
    panic!({
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
