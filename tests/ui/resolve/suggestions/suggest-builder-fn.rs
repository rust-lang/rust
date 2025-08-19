// Tests that we suggest the right alternatives when
// a builder method cannot be resolved

use std::net::TcpStream;

trait SomeTrait {}

struct Foo<T> {
    v : T
}

impl<T: SomeTrait + Default> Foo<T> {
    // Should not be suggested if constraint on the impl not met
    fn new() -> Self {
        Self { v: T::default() }
    }
}

struct Bar;

impl Bar {
    // Should be suggested
    fn build() -> Self {
        Self {}
    }

    // Method with self can't be a builder.
    // Should not be suggested
    fn build_with_self(&self) -> Self {
        Self {}
    }
}

mod SomeMod {
    use crate::Bar;

    impl Bar {
        // Public method. Should be suggested
        pub fn build_public() -> Self {
            Self {}
        }

        // Private method. Should not be suggested
        fn build_private() -> Self {
            Self {}
        }
    }
}

fn main() {
   // `new` not found on `TcpStream` and `connect` should be suggested
   let _stream = TcpStream::new();
   //~^ ERROR no function or associated item named `new` found

    // Although `new` is found on `<impl Foo<T>>` it should not be
    // suggested because `u8` does not meet the `T: SomeTrait` constraint
    let _foo = Foo::<u8>::new();
   //~^ ERROR the function or associated item `new` exists for struct `Foo<u8>`, but its trait bounds were not satisfied

   // Should suggest only `<impl Bar>::build()` and `SomeMod::<impl Bar>::build_public()`.
   // Other methods should not suggested because they are private or are not a builder
    let _bar = Bar::new();
   //~^ ERROR no function or associated item named `new` found
}
