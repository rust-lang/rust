fn main() {
    let x = String::new();
    foo(x.clone()); //~ ERROR mismatched types
}

fn foo(_: &str) {}
