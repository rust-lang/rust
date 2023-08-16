// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn main() {
    let _x = match true {
        false => 0,
        true => panic!(),
    };
}
