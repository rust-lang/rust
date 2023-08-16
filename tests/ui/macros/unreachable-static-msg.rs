// run-fail
//@error-in-other-file:internal error: entered unreachable code: uhoh
//@ignore-target-emscripten no processes

fn main() {
    unreachable!("uhoh")
}
