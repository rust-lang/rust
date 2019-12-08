const fn foo() {
    loop {} //~ ERROR `loop` is not allowed in a `const fn`
}

fn main() {}
