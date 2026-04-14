//! ThingOS-specific extensions to primitives in the [`std::process`] module.

#![stable(feature = "os_thingos", since = "1.0.0")]

use crate::process;

/// OS-specific extensions to [`process::Child`].
#[stable(feature = "os_thingos", since = "1.0.0")]
pub trait ChildExt {}

#[stable(feature = "os_thingos", since = "1.0.0")]
impl ChildExt for process::Child {}

/// OS-specific extensions to [`process::Command`].
#[stable(feature = "os_thingos", since = "1.0.0")]
pub trait CommandExt {}

#[stable(feature = "os_thingos", since = "1.0.0")]
impl CommandExt for process::Command {}

/// OS-specific extensions to [`process::ExitStatus`].
#[stable(feature = "os_thingos", since = "1.0.0")]
pub trait ExitStatusExt {}

#[stable(feature = "os_thingos", since = "1.0.0")]
impl ExitStatusExt for process::ExitStatus {}

/// OS-specific extensions to [`process::ExitStatusError`].
#[stable(feature = "os_thingos", since = "1.0.0")]
pub trait ExitStatusErrorExt {}

#[stable(feature = "os_thingos", since = "1.0.0")]
impl ExitStatusErrorExt for process::ExitStatusError {}
