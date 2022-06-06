fn foo() {
    let x: usize<foo>; //~ ERROR const arguments are not allowed on this type
}

fn main() {}
