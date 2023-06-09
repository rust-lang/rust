//
// run-pass
//
// FIXME(#54366) - We probably shouldn't allow #[thread_local] static mut to get a 'static lifetime.

#![feature(thread_local)]

#[thread_local]
static mut X1: u64 = 0;

struct S1 {
    a: &'static mut u64,
}

impl S1 {
    fn new(_x: u64) -> S1 {
        S1 {
            a: unsafe { &mut X1 },
        }
    }
}

fn main() {
    S1::new(0).a;
}
