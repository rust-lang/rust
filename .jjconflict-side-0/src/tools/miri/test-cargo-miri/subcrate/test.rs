use std::env;
use std::path::PathBuf;

use byteorder::{ByteOrder, LittleEndian};

fn main() {
    println!("subcrate testing");

    fn host_to_target_path(path: String) -> PathBuf {
        use std::ffi::{CStr, CString, c_char};

        let path = CString::new(path).unwrap();
        let mut out = Vec::with_capacity(1024);

        unsafe {
            extern "Rust" {
                fn miri_host_to_target_path(
                    path: *const c_char,
                    out: *mut c_char,
                    out_size: usize,
                ) -> usize;
            }
            let ret = miri_host_to_target_path(path.as_ptr(), out.as_mut_ptr(), out.capacity());
            assert_eq!(ret, 0);
            let out = CStr::from_ptr(out.as_ptr()).to_str().unwrap();
            PathBuf::from(out)
        }
    }

    // CWD should be crate root.
    // We have to normalize slashes, as the env var might be set for a different target's conventions.
    let env_dir = env::current_dir().unwrap();
    let crate_dir = host_to_target_path(env::var("CARGO_MANIFEST_DIR").unwrap());
    assert_eq!(env_dir, crate_dir);

    // Make sure we can call dev-dependencies.
    let _n = <LittleEndian as ByteOrder>::read_u32(&[1, 2, 3, 4]);
}
