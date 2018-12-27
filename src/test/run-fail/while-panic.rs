#![allow(while_true)]

// error-pattern:giraffe
fn main() {
    panic!({
        while true {
            panic!("giraffe")
        }
        "clandestine"
    });
}
