#![allow(unused_crate_dependencies)]

use rustc_thread_pool::join;

#[test]
#[should_panic(expected = "should panic")]
fn simple_panic() {
    join(|| {}, || panic!("should panic"));
}
