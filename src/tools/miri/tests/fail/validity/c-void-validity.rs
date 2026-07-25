use core::ffi::c_void;

fn main() {
    let mut b: u8 = 0;
    let p: *const c_void = (&raw mut b).cast_const().cast();
    let _r: &c_void = unsafe { &*p }; //~ERROR: constructing invalid value
}
