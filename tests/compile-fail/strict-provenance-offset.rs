// compile-flags: -Zmiri-strict-provenance
// error-pattern: not a valid pointer

fn main() {
    let x = 22;
    let ptr = &x as *const _ as *const u8;
    let roundtrip = ptr as usize as *const u8;
    let _ = unsafe { roundtrip.offset(1) };
}
