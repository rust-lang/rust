// run-fail
//@error-in-other-file:quux
//@ignore-target-emscripten no processes

fn f() -> ! {
    panic!("quux")
}
fn g() -> isize {
    match f() {
        true => 1,
        false => 0,
    }
}
fn main() {
    g();
}
