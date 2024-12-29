//@ run-fail
//@ check-run-results:quux
//@ ignore-emscripten no processes

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
