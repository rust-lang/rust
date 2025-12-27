#![unstable(feature = "windows_unix_domain_sockets", issue = "56533")]
mod addr;
mod listener;
mod stream;
pub use addr::*;
pub use listener::*;
pub use stream::*;

use crate::io;
fn not_cvt(value: i32) -> io::Result<()> {
    if value == 0 {
        Ok(())
    } else {
        return Err(io::Error::last_os_error());
    }
}
