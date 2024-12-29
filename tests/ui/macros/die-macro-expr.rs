//@ run-fail
//@ check-run-results:test
//@ ignore-emscripten no processes

fn main() {
    let __isize: isize = panic!("test");
}
