//@compile-flags: -Zmiri-disable-alignment-check -Cdebug-assertions=yes

fn main() {
    let mut x = [0u32; 2];
    let ptr: *mut u8 = x.as_mut_ptr().cast::<u8>();
    unsafe {
        *(ptr.add(1).cast::<u32>()) = 42;
    }
}
