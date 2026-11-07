//! Intrinsics for compile time operations

/// Read a file from the system at comptime time, and returns its bytes.
///
/// Returns [Option::None] if the file cannot be read. This includes when the file does not exist, or hasn't been
/// made available at comptile time by the compiler.
#[rustc_nounwind]
#[rustc_intrinsic]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_comptime]
pub fn comptime_fetch_file(file_path: &str) -> Option<&'static [u8]>;
