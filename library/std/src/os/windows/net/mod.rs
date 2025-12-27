#![unstable(feature = "windows_unix_domain_sockets", issue = "56533")]
mod addr;
mod listener;
mod stream;
pub use addr::*;
pub use listener::*;
pub use stream::*;

use crate::io;
use crate::sys::IsZero;
fn not_cvt(value: impl IsZero) -> io::Result<()> {
    if value.is_zero() {
        Ok(())
    } else {
        return Err(io::Error::last_os_error());
    }
}
