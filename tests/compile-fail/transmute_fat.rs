// This should fail even without validation
// compile-flags: -Zmir-emit-validate=0

fn main() {
    #[cfg(target_pointer_width="64")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], [u8; 16]>(&[1u8])
    };
    #[cfg(target_pointer_width="32")]
    let bad = unsafe {
        std::mem::transmute::<&[u8], [u8; 8]>(&[1u8])
    };
    let _ = bad[0] + bad[bad.len()-1]; //~ ERROR constant evaluation error
    //~^ NOTE a raw memory access tried to access part of a pointer value as raw bytes
}
