//! MVP for compile-type only operations like reading files

/// Read a file at compile time
#[unstable(feature = "type_info", issue = "146922")]
#[rustc_const_unstable(feature = "type_info", issue = "146922")]
#[rustc_comptime]
pub fn read_file(path: &str) -> Option<&'static [u8]> {
    crate::intrinsics::comptime::comptime_fetch_file(path)
}
