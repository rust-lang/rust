// https://github.com/rust-lang/rust/issues/61106
fn main() {
    let x = String::new();
    foo(x.clone()); //~ ERROR mismatched types
}

fn foo(_: &str) {}
