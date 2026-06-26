//@ check-pass

enum Void {}
fn main() {
    let a: Option<Void> = None;
    let None = a;
}
