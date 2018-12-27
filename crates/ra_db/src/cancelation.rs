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
//! In this situation, we want to react to modification as quckly as possible.
//! At the same time, in-progress results are not very interesting, because they
//! are invalidated by the edit anyway. So, we first cancel all in-flight
//! requests, and then apply modification knowing that it won't intrfere with
//! any background processing (this bit is handled by salsa, see
//! `BaseDatabase::check_canceled` method).

use std::{
    cmp,
    hash::{Hash, Hasher},
    sync::Arc,
};

use backtrace::Backtrace;
use parking_lot::Mutex;

/// An "error" signifing that the operation was canceled.
#[derive(Clone)]
pub struct Canceled {
    backtrace: Arc<Mutex<Backtrace>>,
}

pub type Cancelable<T> = Result<T, Canceled>;

impl Canceled {
    pub(crate) fn new() -> Canceled {
        let bt = Backtrace::new_unresolved();
        Canceled {
            backtrace: Arc::new(Mutex::new(bt)),
        }
    }
}

impl std::fmt::Display for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("canceled")
    }
}

impl std::fmt::Debug for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut bt = self.backtrace.lock();
        let bt: &mut Backtrace = &mut *bt;
        bt.resolve();
        write!(fmt, "canceled at:\n{:?}", bt)
    }
}

impl std::error::Error for Canceled {}

impl PartialEq for Canceled {
    fn eq(&self, _: &Canceled) -> bool {
        true
    }
}

impl Eq for Canceled {}

impl Hash for Canceled {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        ().hash(hasher)
    }
}

impl cmp::Ord for Canceled {
    fn cmp(&self, _: &Canceled) -> cmp::Ordering {
        cmp::Ordering::Equal
    }
}

impl cmp::PartialOrd for Canceled {
    fn partial_cmp(&self, other: &Canceled) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
