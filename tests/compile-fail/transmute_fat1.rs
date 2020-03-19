// This should fail even without validation
// compile-flags: -Zmiri-disable-validation

fn main() {
    #[cfg(target_pointer_width="64")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], [u8; 16]>(&[1u8])
    };
    #[cfg(target_pointer_width="32")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], [u8; 8]>(&[1u8])
    };
    let _val = bad[0] + bad[bad.len()-1]; //~ ERROR unable to turn pointer into raw bytes
}
