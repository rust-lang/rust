struct Foo {
    field1: i32,
    field1: i32,
    //~^ ERROR field `field1` is already declared [E0124]
}

fn main() {
}
