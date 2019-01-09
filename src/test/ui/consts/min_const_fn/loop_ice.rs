const fn foo() {
    loop {} //~ ERROR loops are not allowed in const fn
}

fn main() {}
