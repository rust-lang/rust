//@compile-flags: -Zmiri-num-cpus=1024
// FIXME: Disabled on FreeBSD due to <https://github.com/rust-lang/miri/issues/5344>
//@ignore-target: freebsd

use std::num::NonZero;
use std::thread::available_parallelism;

fn main() {
    assert_eq!(available_parallelism().unwrap(), NonZero::new(1024).unwrap());
}
