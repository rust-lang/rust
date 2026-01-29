//@ run-pass

fn foo() {}
fn bar() {}

pub fn main() {
    let _ = [bar, foo];
}
