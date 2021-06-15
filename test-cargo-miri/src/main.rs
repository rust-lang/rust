use byteorder::{BigEndian, ByteOrder};
use std::env;
#[cfg(unix)]
use std::io::{self, BufRead};

fn main() {
    // Check env var set by `build.rs`.
    assert_eq!(env!("MIRITESTVAR"), "testval");

    // Exercise external crate, printing to stdout.
    let buf = &[1,2,3,4];
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
        // CWD should be crate root.
        // We have to normalize slashes, as the env var might be set for a different target's conventions.
        let env_dir = env::current_dir().unwrap();
        let env_dir = env_dir.to_string_lossy().replace("\\", "/");
        let crate_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
        let crate_dir = crate_dir.to_string_lossy().replace("\\", "/");
        assert_eq!(env_dir, crate_dir);

        #[cfg(unix)]
        for line in io::stdin().lock().lines() {
            let num: i32 = line.unwrap().parse().unwrap();
            println!("{}", 2*num);
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
    use rand::{Rng, SeedableRng};

    // Make sure in-crate tests with dev-dependencies work
    #[test]
    fn rng() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xcafebeef);
        let x: u32 = rng.gen();
        let y: usize = rng.gen();
        let z: u128 = rng.gen();
        assert_ne!(x as usize, y);
        assert_ne!(y as u128, z);
    }

    #[test]
    fn exported_symbol() {
        extern crate cargo_miri_test;
        extern crate exported_symbol;
        extern crate issue_rust_86261;
        // Test calling exported symbols in (transitive) dependencies.
        // Repeat calls to make sure the `Instance` cache is not broken.
        for _ in 0..3 {
            extern "Rust" {
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
