fn main() {
    let foo = "bar";
    let x = foo("baz");
    //~^ ERROR: expected function, found `&str`
}

fn foo(file: &str) -> bool {
    true
}
