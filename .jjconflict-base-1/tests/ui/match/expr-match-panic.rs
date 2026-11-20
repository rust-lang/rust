//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

fn main() {
    let _x = match true {
        false => 0,
        true => panic!(),
    };
}
