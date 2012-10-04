//! Logging

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast::transmute;

#[nolink]
extern mod rustrt {
    fn rust_log_console_on();
    fn rust_log_console_off();
    fn rust_log_str(level: u32, string: *libc::c_char, size: libc::size_t);
}

/// Turns on logging to stdout globally
pub fn console_on() {
    rustrt::rust_log_console_on();
}

/**
 * Turns off logging to stdout globally
 *
 * Turns off the console unless the user has overridden the
 * runtime environment's logging spec, e.g. by setting
 * the RUST_LOG environment variable
 */
pub fn console_off() {
    rustrt::rust_log_console_off();
}

#[cfg(notest)]
#[lang="log_type"]
pub fn log_type<T>(level: u32, object: &T) {
    let bytes = do io::with_bytes_writer() |writer| {
        repr::write_repr(writer, object);
    };
    unsafe {
        let len = bytes.len() as libc::size_t;
        rustrt::rust_log_str(level, transmute(vec::raw::to_ptr(bytes)), len);
    }
}

