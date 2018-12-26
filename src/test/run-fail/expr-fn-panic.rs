// error-pattern:explicit panic

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
