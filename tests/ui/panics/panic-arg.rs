// run-fail
//@error-in-other-file:woe
//@ignore-target-emscripten no processes

fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
