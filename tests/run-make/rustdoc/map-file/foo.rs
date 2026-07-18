pub use hidden::Bar;
pub use private::Quz;

mod private {
    pub struct Quz;
}

#[doc(hidden)]
pub mod hidden {
    pub struct Bar;
}

#[macro_export]
macro_rules! foo {
    () => {};
}
