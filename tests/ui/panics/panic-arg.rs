// run-fail
// error-pattern:woe
// ignore-emscripten no processes

fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
