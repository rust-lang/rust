mod a {
    pub struct Foo(u32);

    impl Foo {
        pub fn new() -> Foo { Foo(0) }
    }
}

fn main() {
   let y = a::Foo::new();
   y.0; //~ ERROR field `0` of struct `Foo` is private
}
