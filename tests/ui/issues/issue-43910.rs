//@ run-pass
#![deny(unused_variables)]

fn main() {
    #[allow(unused_variables)]
    let x = 12;
}
