enum Foo {
    Bar(B),
    //~^ ERROR cannot find type `B` in this scope [E0412]
}

fn main() {
    let _ = [0; Foo::Bar as usize];
}
