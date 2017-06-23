fn main() {
    let x = &1;
    // Casting down to u8 and back up to a pointer loses too much precision; this must not work.
    let x = x as *const i32 as u8;
    let x = x as *const i32; //~ ERROR: a raw memory access tried to access part of a pointer value as raw bytes
    let _ = unsafe { *x };
}
