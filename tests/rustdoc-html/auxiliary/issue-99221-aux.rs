pub struct Option;
impl Option {
    pub fn unwrap(self) {}
}

mod macros {
    use crate::Option;
    /// [`Option::unwrap`]
    #[macro_export]
    macro_rules! print {
        () => ()
    }
}

mod structs {
    use crate::Option;
    /// [`Option::unwrap`]
    pub struct Print;
}
pub use structs::Print;
