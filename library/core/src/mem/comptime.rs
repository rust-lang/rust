//! MVP for compile-type only operations like reading files

use std::path::Path;

#[unstable(feature = "type_info", issue = "146922")]
#[rustc_const_unstable(feature = "type_info", issue = "146922")]
#[rustc_comptime]
pub fn read_file<P: Into<Path>>(path: P) -> Option<&'static [u8]> {
    let path = Path::from(path).as_os_str().as_encoded_bytes();
    intrinsics::comptime::comptime_read_file(path)
}
