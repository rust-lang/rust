pub struct Bar;

impl Bar {
    pub fn bar(_: u8) -> hidden::Hidden {
        hidden::Hidden
    }
}

#[doc(hidden)]
pub mod hidden {
    pub struct Hidden;
}
