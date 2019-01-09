//! Utility types to support cancellation.
//!
//! In a typical IDE use-case, requests and modification happen concurrently, as
//! in the following scenario:
//!
//!   * user types a character,
//!   * a syntax highlighting process is started
//!   * user types next character, while syntax highlighting *is still in
//!     progress*.
//!
//! In this situation, we want to react to modification as quickly as possible.
//! At the same time, in-progress results are not very interesting, because they
//! are invalidated by the edit anyway. So, we first cancel all in-flight
//! requests, and then apply modification knowing that it won't interfere with
//! any background processing (this bit is handled by salsa, see the
//! `BaseDatabase::check_canceled` method).

/// An "error" signifing that the operation was canceled.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Canceled {
    _private: (),
}

pub type Cancelable<T> = Result<T, Canceled>;

impl Canceled {
    pub(crate) fn new() -> Canceled {
        Canceled { _private: () }
    }
}

impl std::fmt::Display for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("canceled")
    }
}

impl std::fmt::Debug for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "Canceled")
    }
}

impl std::error::Error for Canceled {}
