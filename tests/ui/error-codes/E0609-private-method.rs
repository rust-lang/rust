// This error is an E0609 and *not* an E0615 because the fact that the method exists is not
// relevant.
mod foo {
    pub struct Foo {
        x: u32,
    }

    impl Foo {
        fn method(&self) {}
    }
}

fn main() {
    let f = foo::Foo { x: 0 };
    f.method; //~ ERROR E0609
}
