// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn main() {
    let _x = if false {
        0
    } else if true {
        panic!()
    } else {
        10
    };
}
