//@ run-fail
//@ check-run-results:woe
//@ ignore-emscripten no processes

fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
