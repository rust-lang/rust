#![allow(while_true)]

// error-pattern:quux
fn main() {
    let _x: isize = {
        while true {
            panic!("quux");
        }
        8
    };
}
