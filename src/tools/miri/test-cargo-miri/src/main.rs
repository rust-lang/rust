use std::env;
#[cfg(unix)]
use std::io::{self, BufRead};
use std::path::PathBuf;

use byteorder::{BigEndian, ByteOrder};

fn main() {
    // Check env var set by `build.rs`.
    assert_eq!(env!("MIRITESTVAR"), "testval");

    // Exercise external crate, printing to stdout.
    let buf = &[1, 2, 3, 4];
    let n = <BigEndian as ByteOrder>::read_u32(buf);
    assert_eq!(n, 0x01020304);
    println!("{:#010x}", n);

    // Access program arguments, printing to stderr.
    for arg in std::env::args() {
        eprintln!("{}", arg);
    }

    // If there were no arguments, access stdin and test working dir.
    // (We rely on the test runner to always disable isolation when passing no arguments.)
    if std::env::args().len() <= 1 {
        fn host_to_target_path(path: String) -> PathBuf {
            use std::ffi::{CStr, CString, c_char};

            let path = CString::new(path).unwrap();
            let mut out = Vec::with_capacity(1024);

            unsafe {
                unsafe extern "Rust" {
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
        let env_dir = env::current_dir().unwrap();
        let crate_dir = host_to_target_path(env::var("CARGO_MANIFEST_DIR").unwrap());
        assert_eq!(env_dir, crate_dir);

        #[cfg(unix)]
        for line in io::stdin().lock().lines() {
            let num: i32 = line.unwrap().parse().unwrap();
            println!("{}", 2 * num);
        }
        // On non-Unix, reading from stdin is not supported. So we hard-code the right answer.
        #[cfg(not(unix))]
        {
            println!("24");
            println!("42");
        }
    }
}

#[cfg(test)]
mod test {
    use byteorder_2::{BigEndian, ByteOrder};

    // Make sure in-crate tests with dev-dependencies work
    #[test]
    fn dev_dependency() {
        let _n = <BigEndian as ByteOrder>::read_u64(&[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn exported_symbol() {
        extern crate cargo_miri_test;
        extern crate exported_symbol;
        extern crate issue_rust_86261;
        // Test calling exported symbols in (transitive) dependencies.
        // Repeat calls to make sure the `Instance` cache is not broken.
        for _ in 0..3 {
            unsafe extern "Rust" {
                fn exported_symbol() -> i32;
                fn assoc_fn_as_exported_symbol() -> i32;
                fn make_true() -> bool;
                fn NoMangleStruct();
                fn no_mangle_generic();
            }
            assert_eq!(unsafe { exported_symbol() }, 123456);
            assert_eq!(unsafe { assoc_fn_as_exported_symbol() }, -123456);
            assert!(unsafe { make_true() });
            unsafe { NoMangleStruct() }
            unsafe { no_mangle_generic() }
        }
    }
}
