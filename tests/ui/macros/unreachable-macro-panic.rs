//@ run-fail
//@ check-run-results:internal error: entered unreachable code
//@ ignore-emscripten no processes

fn main() {
    unreachable!()
}
