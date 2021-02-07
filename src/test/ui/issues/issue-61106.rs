fn main() {
    let x = String::new();
    foo(x.clone()); //~ ERROR arguments to this function are incorrect
}

fn foo(_: &str) {}
