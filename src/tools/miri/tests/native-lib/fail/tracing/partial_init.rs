//@only-target: x86_64-unknown-linux-gnu i686-unknown-linux-gnu
//@compile-flags: -Zmiri-native-lib-enable-tracing

extern "C" {
    fn init_n(n: i32, ptr: *mut u8);
}

fn main() {
    partial_init();
}

// Initialise the first 2 elements of the slice from native code, and check
// that the 3rd is correctly deemed uninit.
fn partial_init() {
    let mut slice = std::mem::MaybeUninit::<[u8; 3]>::uninit();
    let slice_ptr = slice.as_mut_ptr().cast::<u8>();
    unsafe {
        // Initialize the first two elements.
        init_n(2, slice_ptr);
        assert!(*slice_ptr == 0);
        assert!(*slice_ptr.offset(1) == 0);
        // Reading the third is UB!
        let _val = *slice_ptr.offset(2); //~ ERROR: /Undefined Behavior: reading memory.*, but memory is uninitialized/
    }
}
