// `Foo` and `Bar` should not be suggested in diagnostics of dependents

#[doc(hidden)]
pub mod hidden {
    pub struct Foo;
}

pub mod hidden1 {
    #[doc(hidden)]
    pub struct Bar;
}

// `Baz` and `Quux` *should* be suggested in diagnostics of dependents

#[doc(hidden)]
pub mod hidden2 {
    pub struct Baz;
}

pub use hidden2::Baz;

#[doc(hidden)]
pub(crate) mod hidden3 {
    pub struct Quux;
}

pub use hidden3::Quux;

pub trait Marker {}

impl Marker for Option<u32> {}
impl Marker for hidden::Foo {}
impl Marker for hidden1::Bar {}
impl Marker for Baz {}
impl Marker for Quux {}
