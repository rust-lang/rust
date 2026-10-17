macro_rules! a {
    () => {trait A {}; 0}
    //~^ ERROR: expected expression, found keyword `trait`
    //~| ERROR: expected expression, found keyword `trait`
}

fn main() {
    let a = a!();
    a!()
}
