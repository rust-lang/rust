// compile-flags: -Zmiri-strict-provenance
// error-pattern: is a dangling pointer
#![feature(strict_provenance)]

fn main() {
    let x = 22;
    let ptr = &x as *const _ as *const u8;
    let roundtrip = std::ptr::invalid::<u8>(ptr as usize);
    // Not even offsetting this is allowed.
    let _ = unsafe { roundtrip.offset(1) };
}
