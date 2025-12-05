pub mod Mod {
    pub struct Foo {}
    impl Foo {
        pub const BAR: usize = 42;
    }
}

fn foo(_: usize) {}

fn main() {
    foo(Mod::Foo.Bar);
    //~^ ERROR expected value, found
}
