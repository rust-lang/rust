//! Ensure that we call Windows TLS callbacks in the local crate.
//@only-target: windows
//@revisions: with_exit without_exit
// run-native does not work properly. It randomly sometimes calls the hook 3 times. I guess there's
// sometimes another thread appearing from somewhere.

// Printing without touching the Rust runtime.
fn print_safe(msg: &str) {
    use std::io::Write;
    use std::os::windows::io::FromRawHandle;
    use std::{ffi, fs, mem};

    pub const STD_OUTPUT_HANDLE: u32 = 4294967285;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut ffi::c_void;
    }

    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut handle = mem::ManuallyDrop::new(fs::File::from_raw_handle(handle));
        handle.write_all(msg.as_bytes()).unwrap();
    }
}

#[link_section = ".CRT$XLB"]
#[used] // Miri only considers explicitly `#[used]` statics for `lookup_link_section`
pub static CALLBACK: unsafe extern "system" fn(*const (), u32, *const ()) = tls_callback;

unsafe extern "system" fn tls_callback(_h: *const (), _dw_reason: u32, _pv: *const ()) {
    // Do not call into std as it has already been shut down. That means we cannot run natively.
    print_safe("in tls_callback\n");
}

fn main() {
    // Spawn a thread so that we see the hook called twice.
    std::thread::spawn(|| {});
    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
