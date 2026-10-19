//! Ensure that we call Windows TLS callbacks in the local crate.
//@only-target: windows

use core::fmt::Write;

#[path = "../../utils/mod.rs"]
mod utils;

#[link_section = ".CRT$XLB"]
#[used] // Miri only considers explicitly `#[used]` statics for `lookup_link_section`
pub static CALLBACK: unsafe extern "system" fn(*const (), u32, *const ()) = tls_callback;

unsafe extern "system" fn tls_callback(_h: *const (), _dw_reason: u32, _pv: *const ()) {
    // Do not call into std as it has already been shut down. That means we cannot run natively.
    writeln!(utils::MiriStderr, "in tls_callback").unwrap();
}

fn main() {
    std::thread::spawn(|| {});
}
