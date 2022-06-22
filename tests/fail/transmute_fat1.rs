// error-pattern: type validation failed: encountered a pointer
// normalize-stderr-test: "\[u8; (08|16)\]" -> "$$ARRAY"

fn main() {
    #[cfg(target_pointer_width = "64")]
    let bad = unsafe { std::mem::transmute::<&[u8], [u8; 16]>(&[1u8]) };
    #[cfg(target_pointer_width = "32")]
    let bad = unsafe { std::mem::transmute::<&[u8], [u8; 08]>(&[1u8]) };
    let _val = bad[0] + bad[bad.len() - 1];
}
