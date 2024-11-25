//@ run-fail
//@ check-run-results:internal error: entered unreachable code: uhoh
//@ ignore-emscripten no processes

fn main() {
    unreachable!("uhoh")
}
