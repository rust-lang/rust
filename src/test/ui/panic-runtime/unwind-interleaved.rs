// run-fail
// error-pattern:explicit panic

fn a() {}

fn b() {
    panic!();
}

fn main() {
    let _x = vec![0];
    a();
    let _y = vec![0];
    b();
}
