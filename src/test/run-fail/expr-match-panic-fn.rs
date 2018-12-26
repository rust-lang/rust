// error-pattern:explicit panic

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
