//@ run-fail
//@ error-pattern:test
//@ ignore-emscripten no processes

fn main() {
    let __isize: isize = panic!("test");
}
