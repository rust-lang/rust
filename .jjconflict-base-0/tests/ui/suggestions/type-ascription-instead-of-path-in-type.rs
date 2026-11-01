enum A {
    B,
}

fn main() {
    let _: Vec<A:B> = A::B;
    //~^ ERROR cannot find trait `B` in this scope
    //~| HELP you might have meant to write a path instead of an associated type bound
}
