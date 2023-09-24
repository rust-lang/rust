enum Foo {
    AFoo,
    BFoo,
    CFoo,
    DFoo,
}
enum Foo2 {
    //~^ ERROR: all variants have the same postfix
    AFoo,
    BFoo,
    CFoo,
    DFoo,
    EFoo,
}

fn main() {}
