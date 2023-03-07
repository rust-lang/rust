// run-pass
// Tests equality between supertype and subtype of a function
// See the issue #91636
fn foo(_a: &str) {}

fn main() {
    let x = foo as fn(&'static str);
    let _ = x == foo;
}
