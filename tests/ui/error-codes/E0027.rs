struct Dog {
    name: String,
    age: u32,
}

pub mod foo {
    pub struct Bar {
        pub field1: (),
        field2: (),
    }
    impl Bar {
        pub fn new() -> Self {
            Bar {
                field1: (),
                field2: (),
            }
        }
    }
    pub struct Baz {
        pub field1: (),
        pub field2: (),
        pub field3: (),
        field4: (),
    }
    impl Baz {
        pub fn new() -> Self {
            Baz {
                field1: (),
                field2: (),
                field3: (),
                field4: (),
            }
        }
    }
}


fn main() {
    let d = Dog { name: "Rusty".to_string(), age: 8 };

    match d {
        Dog { age: x } => {} //~ ERROR pattern does not mention field `name`
    }
    match d {
        // trailing comma
        Dog { name: x, } => {} //~ ERROR pattern does not mention field `age`
    }
    match d {
        // trailing comma with weird whitespace
        Dog { name: x  , } => {} //~ ERROR pattern does not mention field `age`
    }
    match d {
        Dog {} => {} //~ ERROR pattern does not mention fields `name` and `age`
    }
    match foo::Bar::new() {
        foo::Bar {} => {} //~ ERROR pattern does not mention field `field1` and inaccessible fields
    }
    match foo::Baz::new() {
        foo::Baz {} => {} //~ ERROR pattern does not mention fields `field1`, `field2`, `field3` and inaccessible fields
    }
    match foo::Baz::new() {
        foo::Baz { field1 } => {} //~ ERROR pattern does not mention fields `field2`, `field3` and inaccessible fields
    }
    match foo::Baz::new() {
        foo::Baz { field1, field2, field3 } => {} //~ ERROR pattern requires `..` due to inaccessible fields
    }
}
