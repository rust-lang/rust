fn foo() {
    let x: usize<foo>; //~ ERROR const arguments are not allowed for this type
}

fn main() {}
