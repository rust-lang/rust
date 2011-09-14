// error-pattern:fail

fn a() { }

fn b() { fail; }

fn main() {
    let x = [0];
    a();
    let y = [0];
    b();
}