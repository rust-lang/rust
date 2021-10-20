use crate::fmt;
use crate::panic::assert_info::AssertInfo;

/// Describes the cause of the panic.
#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
#[non_exhaustive]
pub enum PanicDescription<'a> {
    /// Formatted arguments that were passed to the panic.
    Message(&'a fmt::Arguments<'a>),
    /// Information about the assertion that caused the panic.
    AssertInfo(&'a AssertInfo<'a>),
}

#[unstable(
    feature = "panic_internals",
    reason = "internal details of the implementation of the `panic!` and related macros",
    issue = "none"
)]
impl fmt::Display for PanicDescription<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => write!(formatter, "{}", message),
            Self::AssertInfo(info) => write!(formatter, "{}", info),
        }
    }
}
