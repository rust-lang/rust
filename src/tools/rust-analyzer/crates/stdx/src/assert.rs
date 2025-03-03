// Vendored from https://github.com/matklad/always-assert/commit/4cf564eea6fcf18b30c3c3483a611004dc03afbb
//! Recoverable assertions, inspired by [the use of `assert()` in
//! SQLite](https://www.sqlite.org/assert.html).
//!
//! `never!` and `always!` return the actual value of the condition if
//! `debug_assertions` are disabled.
//!
//! Use them when terminating on assertion failure is worse than continuing.
//!
//! One example would be a critical application like a database:
//!
//! ```ignore
//! use stdx::never;
//!
//! fn apply_transaction(&mut self, tx: Transaction) -> Result<(), TransactionAborted> {
//!     let delta = self.compute_delta(&tx);
//!
//!     if never!(!self.check_internal_invariant(&delta)) {
//!         // Ok, something in this transaction messed up our internal state.
//!         // This really shouldn't be happening, and this signifies a bug.
//!         // Luckily, we can recover by just rejecting the transaction.
//!         return abort_transaction(tx);
//!     }
//!     self.commit(delta);
//!     Ok(())
//! }
//! ```
//!
//! Another example is assertions about non-critical functionality in usual apps
//!
//! ```ignore
//! use stdx::never;
//!
//! let english_message = "super app installed!"
//! let mut local_message = localize(english_message);
//! if never!(local_message.is_empty(), "missing localization for {}", english_message) {
//!     // We localized all the messages but this one slipper through the cracks?
//!     // Better to show the english one then than to fail outright;
//!     local_message = english_message;
//! }
//! println!("{}", local_message);
//! ```

/// Asserts that the condition is always true and returns its actual value.
///
/// If the condition is true does nothing and and evaluates to true.
///
/// If the condition is false:
/// * panics if `force` feature or `debug_assertions` are enabled,
/// * logs an error if the `tracing` feature is enabled,
/// * evaluates to false.
///
/// Accepts `format!` style arguments.
#[macro_export]
macro_rules! always {
    ($cond:expr) => {
        $crate::always!($cond, "assertion failed: {}", stringify!($cond))
    };

    ($cond:expr, $fmt:literal $($arg:tt)*) => {{
        let cond = $cond;
        if cfg!(debug_assertions) || $crate::assert::__FORCE {
            assert!(cond, $fmt $($arg)*);
        }
        if !cond {
            $crate::assert::__tracing_error!($fmt $($arg)*);
        }
        cond
    }};
}

/// Asserts that the condition is never true and returns its actual value.
///
/// If the condition is false does nothing and and evaluates to false.
///
/// If the condition is true:
/// * panics if `force` feature or `debug_assertions` are enabled,
/// * logs an error if the `tracing` feature is enabled,
/// * evaluates to true.
///
/// Accepts `format!` style arguments.
///
/// Empty condition is equivalent to false:
///
/// ```ignore
/// never!("oups") ~= unreachable!("oups")
/// ```
#[macro_export]
macro_rules! never {
    (true $($tt:tt)*) => { $crate::never!((true) $($tt)*) };
    (false $($tt:tt)*) => { $crate::never!((false) $($tt)*) };
    () => { $crate::never!("assertion failed: entered unreachable code") };
    ($fmt:literal $(, $($arg:tt)*)?) => {{
        if cfg!(debug_assertions) || $crate::assert::__FORCE {
            unreachable!($fmt $(, $($arg)*)?);
        }
        $crate::assert::__tracing_error!($fmt $(, $($arg)*)?);
    }};

    ($cond:expr) => {{
        let cond = !$crate::always!(!$cond);
        cond
    }};

    ($cond:expr, $fmt:literal $($arg:tt)*) => {{
        let cond = !$crate::always!(!$cond, $fmt $($arg)*);
        cond
    }};
}

#[doc(hidden)]
pub use tracing::error as __tracing_error;

#[doc(hidden)]
pub const __FORCE: bool = cfg!(feature = "force-always-assert");
