#![feature(unchecked_shifts)]

fn main() {
    unsafe {
        let _n = 1i8.unchecked_shl(8);
        //~^ ERROR: overflowing shift by 8 in `unchecked_shl`
    }
}
