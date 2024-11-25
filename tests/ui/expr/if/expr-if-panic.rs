//@ run-fail
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn main() {
    let _x = if false {
        0
    } else if true {
        panic!()
    } else {
        10
    };
}
