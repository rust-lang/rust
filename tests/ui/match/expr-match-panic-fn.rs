// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

fn f() -> ! {
    panic!()
}

fn g() -> isize {
    let x = match true {
        true => f(),
        false => 10,
    };
    return x;
}

fn main() {
    g();
}
