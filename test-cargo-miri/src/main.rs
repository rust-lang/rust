use byteorder::{BigEndian, ByteOrder};
use std::env;
use std::path::PathBuf;
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
    if std::env::args().len() <= 1 {
        let env_dir = env::current_dir().unwrap();
        let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        assert_eq!(env_dir, crate_dir);

        #[cfg(unix)]
        for line in io::stdin().lock().lines() {
            let num: i32 = line.unwrap().parse().unwrap();
            println!("{}", 2*num);
        }
        // On non-Unix, reading from stdin is not support. So we hard-code the right answer.
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
}
