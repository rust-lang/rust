// run-fail
// error-pattern:bad input
// ignore-emscripten no processes

fn main() {
    Some("foo").unwrap_or(panic!("bad input")).to_string();
}
