mod a {
    pub struct Foo {
        x: u32,
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }
    }
}

fn main() {
    let f = a::Foo::new();
    f.x; //~ ERROR E0616
}
