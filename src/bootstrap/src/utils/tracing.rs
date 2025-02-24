//! Wrapper macros for `tracing` macros to avoid having to write `cfg(feature = "tracing")`-gated
//! `debug!`/`trace!` everytime, e.g.
//!
//! ```rust,ignore (example)
//! #[cfg(feature = "tracing")]
//! trace!("...");
//! ```
//!
//! When `feature = "tracing"` is inactive, these macros expand to nothing.

#[macro_export]
macro_rules! trace {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::trace!($($tokens)*)
    }
}

#[macro_export]
macro_rules! debug {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::debug!($($tokens)*)
    }
}

#[macro_export]
macro_rules! warn {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::warn!($($tokens)*)
    }
}

#[macro_export]
macro_rules! info {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::info!($($tokens)*)
    }
}

#[macro_export]
macro_rules! error {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::error!($($tokens)*)
    }
}

#[macro_export]
macro_rules! trace_cmd {
    ($cmd:expr) => {
        {
            use $crate::utils::exec::FormatShortCmd;

            ::tracing::span!(
                target: "COMMAND",
                ::tracing::Level::TRACE,
                "executing command",
                cmd = $cmd.format_short_cmd(),
                full_cmd = ?$cmd
            ).entered()
        }
    };
}
