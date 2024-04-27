// Test to ensure private traits are inaccessible with UFCS angle-bracket syntax.

mod foo {
    trait Bar {
        fn baz() {}
    }

    impl Bar for i32 {}
}

fn main() {
    <i32 as ::foo::Bar>::baz(); //~ERROR trait `Bar` is private
}
