// Regression test for https://github.com/rust-lang/rust/issues/122581
// This used to ICE, because the union was unsized and the pointer casting code
// assumed that non-struct ADTs must be sized.

union Union {
    val: std::mem::ManuallyDrop<[u8]>,
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

fn cast(ptr: *const ()) -> *const Union {
    ptr as _
}

fn main() {}
