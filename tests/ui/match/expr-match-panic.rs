// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

fn main() {
    let _x = match true {
        false => 0,
        true => panic!(),
    };
}
