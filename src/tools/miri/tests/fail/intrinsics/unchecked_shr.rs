#![feature(unchecked_shifts)]

fn main() {
    unsafe {
        let _n = 1i64.unchecked_shr(64);
        //~^ ERROR: overflowing shift by 64 in `unchecked_shr`
    }
}
