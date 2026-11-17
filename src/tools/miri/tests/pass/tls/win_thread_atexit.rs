//! Ensure that we call Windows TLS callbacks in the local crate.
//@only-target: windows
//@revisions: with_exit without_exit
// run-native does not work properly. It randomly sometimes calls the hook 3 times. I guess there's
// sometimes another thread appearing from somewhere.

#[link_section = ".CRT$XLB"]
#[used] // Miri only considers explicitly `#[used]` statics for `lookup_link_section`
pub static CALLBACK: unsafe extern "system" fn(*const (), u32, *const ()) = tls_callback;

unsafe extern "system" fn tls_callback(_h: *const (), _dw_reason: u32, _pv: *const ()) {
    // Do not call into std as it has already been shut down. That means we cannot run natively.
    eprintln!("in tls_callback");
}

fn main() {
    // Spawn a thread so that we see the hook called twice.
    std::thread::spawn(|| {});
    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
