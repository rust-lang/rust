//@ run-fail
//@ check-run-results
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
