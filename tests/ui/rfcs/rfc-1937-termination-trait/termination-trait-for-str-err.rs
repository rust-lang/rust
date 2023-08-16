// run-fail
//@error-in-other-file: An error message for you
// failure-status: 1
//@ignore-target-emscripten no processes

fn main() -> Result<(), &'static str> {
    Err("An error message for you")
}
