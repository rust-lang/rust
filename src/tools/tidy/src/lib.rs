//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use walk::{filter_dirs, walk, walk_many, walk_no_read};

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The expression that failed
/// * The error itself
/// * (optionally) a path connected to the error (e.g. failure to open a file)
#[macro_export]
macro_rules! t {
    ($e:expr, $p:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed on {} with {}", stringify!($e), ($p).display(), e),
        }
    };

    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
}

macro_rules! tidy_error {
    ($bad:expr, $fmt:expr) => ({
        *$bad = true;
        eprint!("tidy error: ");
        eprintln!($fmt);
    });
    ($bad:expr, $fmt:expr, $($arg:tt)*) => ({
        *$bad = true;
        eprint!("tidy error: ");
        eprintln!($fmt, $($arg)*);
    });
}

pub mod bins;
pub mod debug_artifacts;
pub mod deps;
pub mod edition;
pub mod error_codes_check;
pub mod errors;
pub mod extdeps;
pub mod features;
pub mod pal;
pub mod primitive_docs;
pub mod style;
pub mod target_specific_tests;
pub mod ui_tests;
pub mod unit_tests;
pub mod unstable_book;
pub mod walk;
