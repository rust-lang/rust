// Stacked Borrows disallows this becuase the reference is never cast to a raw pointer.
// compile-flags: -Zmiri-disable-stacked-borrows

fn main() {
    // If we are careful, we can exploit data layout...
    // This is a tricky case since we are transmuting a ScalarPair type to a non-ScalarPair type.
    let raw = unsafe { std::mem::transmute::<&[u8], [*const u8; 2]>(&[42]) };
    let ptr: *const u8 = unsafe { std::mem::transmute_copy(&raw) };
    assert_eq!(unsafe { *ptr }, 42);
}
