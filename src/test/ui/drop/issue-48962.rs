// run-pass
#![allow(unused_must_use)]
// Test that we are able to reinitialize box with moved referent
static mut ORDER: [usize; 3] = [0, 0, 0];
static mut INDEX: usize = 0;

struct Dropee (usize);

impl Drop for Dropee {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = self.0;
            INDEX = INDEX + 1;
        }
    }
}

fn add_sentintel() {
    unsafe {
        ORDER[INDEX] = 2;
        INDEX = INDEX + 1;
    }
}

fn main() {
    let mut x = Box::new(Dropee(1));
    *x;  // move out from `*x`
    add_sentintel();
    *x = Dropee(3); // re-initialize `*x`
    {x}; // drop value
    unsafe {
        assert_eq!(ORDER, [1, 2, 3]);
    }
}
