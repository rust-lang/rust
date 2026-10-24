//@compile-flags: -Zmiri-num-cpus=1024

use std::num::NonZero;
use std::thread::available_parallelism;

fn main() {
    assert_eq!(available_parallelism().unwrap(), NonZero::new(1024).unwrap());
}
