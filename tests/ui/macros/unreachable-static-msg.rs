//@ run-fail
//@ error-pattern:internal error: entered unreachable code: uhoh
//@ ignore-emscripten no processes

fn main() {
    unreachable!("uhoh")
}
