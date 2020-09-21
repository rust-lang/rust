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
            //~^ ERROR thread-local variable borrowed past end of function
        }
    }
}

fn main() {
    S1::new(0).a;
}
