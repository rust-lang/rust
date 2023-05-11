enum SomeEnum {
    E
}

fn main() {
    E { name: "foobar" }; //~ ERROR cannot find struct, variant or union type `E`
}
