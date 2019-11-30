//! VxWorks-specific raw type definitions
#![stable(feature = "metadata_ext", since = "1.1.0")]

use crate::os::raw::c_ulong;

#[stable(feature = "pthread_t", since = "1.8.0")]
pub type pthread_t = c_ulong;
