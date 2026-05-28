#![feature(core_intrinsics)]

// Test that these intrinsics work. Their behavior should be a no-op.

fn main() {
    static X: [u8; 8] = [0; 8];

    ::std::intrinsics::prefetch_read_data::<_, 1>(::std::ptr::null::<u8>());
    ::std::intrinsics::prefetch_read_data::<_, 2>(::std::ptr::dangling::<u8>());
    ::std::intrinsics::prefetch_read_data::<_, 3>(X.as_ptr());

    ::std::intrinsics::prefetch_write_data::<_, 1>(::std::ptr::null::<u8>());
    ::std::intrinsics::prefetch_write_data::<_, 2>(::std::ptr::dangling::<u8>());
    ::std::intrinsics::prefetch_write_data::<_, 3>(X.as_ptr());

    ::std::intrinsics::prefetch_read_instruction::<_, 1>(::std::ptr::null::<u8>());
    ::std::intrinsics::prefetch_read_instruction::<_, 2>(::std::ptr::dangling::<u8>());
    ::std::intrinsics::prefetch_read_instruction::<_, 3>(X.as_ptr());

    ::std::intrinsics::prefetch_write_instruction::<_, 1>(::std::ptr::null::<u8>());
    ::std::intrinsics::prefetch_write_instruction::<_, 2>(::std::ptr::dangling::<u8>());
    ::std::intrinsics::prefetch_write_instruction::<_, 3>(X.as_ptr());
}
