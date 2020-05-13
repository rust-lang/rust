#![stable(feature = "core", since = "1.6.0")]
#![allow(missing_docs)]
#[cfg(not(bootstrap))]
#[stable(feature = "core", since = "1.6.0")]
#[lang = "manageable_contents"]
/// This trait can be implemented on types where it is safe to allow the allow the collector to
/// free its memory and omit the drop method. This prevents the need to register a finalizer when
/// managed by the Gc which is expensive.
pub trait ManageableContents {}
