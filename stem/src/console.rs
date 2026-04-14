//! Console output - delegates to platform abstraction layer.

use crate::pal;
use core::fmt;

pub fn log(level: usize, args: fmt::Arguments) {
    let pal_level = match level {
        0 => pal::log::Level::Error,
        1 => pal::log::Level::Warn,
        3 => pal::log::Level::Info,
        4 => pal::log::Level::Debug,
        5 => pal::log::Level::Trace,
        _ => pal::log::Level::Info,
    };
    pal::log::write(pal_level, args);
}

pub fn log_with_provenance(level: usize, provenance: &str, args: fmt::Arguments) {
    let pal_level = match level {
        0 => pal::log::Level::Error,
        1 => pal::log::Level::Warn,
        3 => pal::log::Level::Info,
        4 => pal::log::Level::Debug,
        5 => pal::log::Level::Trace,
        _ => pal::log::Level::Info,
    };
    pal::log::write_with_provenance(pal_level, provenance, args);
}

pub fn print(args: fmt::Arguments) {
    log(3, args); // Default to Info
}
