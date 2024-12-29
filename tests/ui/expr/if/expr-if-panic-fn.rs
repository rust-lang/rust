//@ run-fail
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn f() -> ! {
    panic!()
}

fn g() -> isize {
    let x = if true {
        f()
    } else {
        10
    };
    return x;
}

fn main() {
    g();
}
