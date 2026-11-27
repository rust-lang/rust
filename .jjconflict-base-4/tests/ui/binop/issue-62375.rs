enum A {
    Value(())
}

fn main() {
    let a = A::Value(());
    a == A::Value;
    //~^ ERROR binary operation `==` cannot be applied to type `A`
}
