//! MVP for compile-type only operations like reading files

/// Read a file at compile time. Returns its content if it exists.
///
/// This returns [Option::None] if the file does not exist.
///
/// # Compile errors
/// This function will fail to compile in the following conditions:
/// - file has not been made available at compile time by the user
/// - an actual error happens when the compiler tries
///   to read the file (e.g PermissionDenied, IsADirectory)
#[unstable(feature = "comptime_ops_fs", issue = "165159")]
#[rustc_const_unstable(feature = "comptime_ops_fs", issue = "165159")]
#[rustc_comptime]
#[track_caller]
pub fn read_file(path: &str) -> Option<&'static [u8]> {
    crate::intrinsics::comptime::comptime_fetch_file(path)
}

/// Read the list of files that are accessible at compile time
#[unstable(feature = "comptime_ops_fs", issue = "165159")]
#[rustc_const_unstable(feature = "comptime_ops_fs", issue = "165159")]
#[rustc_comptime]
#[track_caller]
pub fn list_files() -> &'static [&'static str] {
    crate::intrinsics::comptime::comptime_list_files()
}
