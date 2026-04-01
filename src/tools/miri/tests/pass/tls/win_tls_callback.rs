//! Ensure that we call Windows TLS callbacks in the local crate.
//@only-target: windows
// Calling eprintln in the callback seems to (re-)initialize some thread-local storage
// and then leak the memory allocated for that. Let's just ignore these leaks,
// that's not what this test is about.
//@compile-flags: -Zmiri-ignore-leaks

#[link_section = ".CRT$XLB"]
#[used] // Miri only considers explicitly `#[used]` statics for `lookup_link_section`
pub static CALLBACK: unsafe extern "system" fn(*const (), u32, *const ()) = tls_callback;

unsafe extern "system" fn tls_callback(_h: *const (), _dw_reason: u32, _pv: *const ()) {
    eprintln!("in tls_callback");
}

fn main() {}
