//! Small helper functions used inside `sys`.
//!
//! If any of these have uses outside of `sys`, please move them to a different
//! module.

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
mod small_c_string;
#[cfg_attr(not(target_os = "windows"), allow(unused))] // Not used on all platforms.
mod wstr;

#[cfg(test)]
mod tests;

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
pub use small_c_string::{run_path_with_cstr, run_with_cstr};
#[cfg_attr(not(target_os = "windows"), allow(unused))] // Not used on all platforms.
pub use wstr::WStrUnits;

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
pub fn ignore_notfound<T>(result: crate::io::Result<T>) -> crate::io::Result<()> {
    match result {
        Err(err) if err.kind() == crate::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Ok(()),
        Err(err) => Err(err),
    }
}
