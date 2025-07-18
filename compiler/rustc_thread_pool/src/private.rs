//! The public parts of this private module are used to create traits
//! that cannot be implemented outside of our own crate. This way we
//! can feel free to extend those traits without worrying about it
//! being a breaking change for other implementations.

/// If this type is pub but not publicly reachable, third parties
/// can't name it and can't implement traits using it.
#[allow(missing_debug_implementations)]
pub struct PrivateMarker;

macro_rules! private_decl {
    () => {
        /// This trait is private; this method exists to make it
        /// impossible to implement outside the crate.
        #[doc(hidden)]
        fn __rayon_private__(&self) -> crate::private::PrivateMarker;
    };
}

macro_rules! private_impl {
    () => {
        fn __rayon_private__(&self) -> crate::private::PrivateMarker {
            crate::private::PrivateMarker
        }
    };
}
