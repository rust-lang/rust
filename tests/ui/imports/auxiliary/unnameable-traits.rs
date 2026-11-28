mod private {
    pub trait Unnameable {
        fn unnameable_method(&self) {}
    }
    impl Unnameable for u8 {}

    pub trait Leaked {
        fn leaked_method(&self) {}
    }
    impl Leaked for u8 {}

    pub trait Reexported {
        fn reexported_method(&self) {}
    }
    impl Reexported for u8 {}
}

pub use private::Reexported;

pub fn leak() -> impl private::Leaked {
    0u8
}
