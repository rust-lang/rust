// compile-flags: -Cmetadata=aux

pub struct Foo;

#[doc(hidden)]
mod bar {
    trait Bar {}

    impl Bar for ::Foo {}
}
