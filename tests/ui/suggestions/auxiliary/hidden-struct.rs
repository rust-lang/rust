#[doc(hidden)]
pub mod hidden {
    pub struct Foo;
}

pub mod hidden1 {
    #[doc(hidden)]
    pub struct Foo;
}


#[doc(hidden)]
pub(crate) mod hidden2 {
    pub struct Bar;
}

pub use hidden2::Bar;
