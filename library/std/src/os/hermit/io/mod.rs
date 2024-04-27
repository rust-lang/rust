#![stable(feature = "os_fd", since = "1.66.0")]

mod net;
#[path = "../../fd/owned.rs"]
mod owned;
#[path = "../../fd/raw.rs"]
mod raw;

// Export the types and traits for the public API.
#[stable(feature = "os_fd", since = "1.66.0")]
pub use owned::*;
#[stable(feature = "os_fd", since = "1.66.0")]
pub use raw::*;
