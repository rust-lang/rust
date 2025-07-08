//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let _x = match true {
        false => 0,
        true => panic!(),
    };
}
