//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

fn main() {
    let _x = if false {
        0
    } else if true {
        panic!()
    } else {
        10
    };
}
