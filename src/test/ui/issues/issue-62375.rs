enum A {
    Value(())
}

fn main() {
    let a = A::Value(());
    a == A::Value;
    //~^ ERROR can't compare
}
