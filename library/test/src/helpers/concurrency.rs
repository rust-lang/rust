//! Helper module which helps to determine amount of threads to be used
//! during tests execution.
use std::env;
use std::thread;

#[allow(deprecated)]
pub fn get_concurrency() -> usize {
    match env::var("RUST_TEST_THREADS") {
        Ok(s) => {
            let opt_n: Option<usize> = s.parse().ok();
            match opt_n {
                Some(n) if n > 0 => n,
                _ => panic!("RUST_TEST_THREADS is `{}`, should be a positive integer.", s),
            }
        }
        Err(..) => thread::available_concurrency().map(|n| n.get()).unwrap_or(1),
    }
}
