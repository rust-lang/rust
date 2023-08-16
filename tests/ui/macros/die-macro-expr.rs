// run-fail
//@error-in-other-file:test
//@ignore-target-emscripten no processes

fn main() {
    let __isize: isize = panic!("test");
}
