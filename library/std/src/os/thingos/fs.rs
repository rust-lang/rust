//! ThingOS-specific extensions to primitives in the [`std::fs`] module.

#![stable(feature = "os_thingos", since = "1.0.0")]

use crate::fs;

/// OS-specific extensions to [`fs::Metadata`].
#[stable(feature = "os_thingos", since = "1.0.0")]
pub trait MetadataExt {}

#[stable(feature = "os_thingos", since = "1.0.0")]
impl MetadataExt for fs::Metadata {}
