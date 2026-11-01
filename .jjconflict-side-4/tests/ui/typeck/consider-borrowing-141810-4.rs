fn baz(x: &String) {}

fn bar() {
    baz({
        String::from("hi") //~ ERROR mismatched types
    });
}

fn main() {
    bar();
}
