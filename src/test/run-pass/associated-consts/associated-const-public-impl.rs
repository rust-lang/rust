// run-pass

mod bar1 {
    pub use self::bar2::Foo;
    mod bar2 {
        pub struct Foo;

        impl Foo {
            pub const ID: i32 = 1;
        }
    }
}

fn main() {
    assert_eq!(1, bar1::Foo::ID);
}
