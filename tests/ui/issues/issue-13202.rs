// run-fail
//@error-in-other-file:bad input
//@ignore-target-emscripten no processes

fn main() {
    Some("foo").unwrap_or(panic!("bad input")).to_string();
}
