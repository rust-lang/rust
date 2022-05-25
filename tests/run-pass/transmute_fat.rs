// Stacked Borrows disallows this becuase the reference is never cast to a raw pointer.
// compile-flags: -Zmiri-disable-stacked-borrows -Zmiri-allow-ptr-int-transmute

fn main() {
    // If we are careful, we can exploit data layout...
    let raw = unsafe {
        std::mem::transmute::<&[u8], [usize; 2]>(&[42])
    };
    let ptr = raw[0] + raw[1];
    // We transmute both ways, to really test allow-ptr-int-transmute.
    let ptr: *const u8 = unsafe { std::mem::transmute(ptr) };
    // The pointer is one-past-the end, but we decrement it into bounds before using it
    assert_eq!(unsafe { *ptr.offset(-1) }, 42);
}
