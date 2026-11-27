pub struct Foo;

pub trait Woof {}
pub trait Bark {}

mod private {
    // should be shown
    impl ::Woof for ::Foo {}

    pub trait Bar {}
    pub struct Wibble;

    // these should not be shown
    impl Bar for ::Foo {}
    impl Bar for Wibble {}
    impl ::Bark for Wibble {}
    impl ::Woof for Wibble {}
}

#[doc(hidden)]
pub mod hidden {
    // should be shown
    impl ::Bark for ::Foo {}

    pub trait Qux {}
    pub struct Wobble;


    // these should only be shown if they're re-exported correctly
    impl Qux for ::Foo {}
    impl Qux for Wobble {}
    impl ::Bark for Wobble {}
    impl ::Woof for Wobble {}
}
