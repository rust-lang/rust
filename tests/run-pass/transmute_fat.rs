// Stacked Borrows disallows this becuase the reference is never cast to a raw pointer.
// compile-flags: -Zmiri-disable-stacked-borrows

fn main() {
    // If we are careful, we can exploit data layout...
    let raw = unsafe {
        std::mem::transmute::<&[u8], [usize; 2]>(&[42])
    };
    let ptr = raw[0] + raw[1];
    let ptr = ptr as *const u8;
    // The pointer is one-past-the end, but we decrement it into bounds before using it
    assert_eq!(unsafe { *ptr.offset(-1) }, 42);
}
