#![feature(core_intrinsics)]

fn main() {
    static X: [u8; 8] = [0; 8];

    unsafe {
        ::std::intrinsics::prefetch_read_data(::std::ptr::null::<u8>(), 1);
        ::std::intrinsics::prefetch_read_data(::std::ptr::dangling::<u8>(), 2);
        ::std::intrinsics::prefetch_read_data(X.as_ptr(), 3);

        ::std::intrinsics::prefetch_write_data(::std::ptr::null::<u8>(), 1);
        ::std::intrinsics::prefetch_write_data(::std::ptr::dangling::<u8>(), 2);
        ::std::intrinsics::prefetch_write_data(X.as_ptr(), 3);

        ::std::intrinsics::prefetch_read_instruction(::std::ptr::null::<u8>(), 1);
        ::std::intrinsics::prefetch_read_instruction(::std::ptr::dangling::<u8>(), 2);
        ::std::intrinsics::prefetch_read_instruction(X.as_ptr(), 3);

        ::std::intrinsics::prefetch_write_instruction(::std::ptr::null::<u8>(), 1);
        ::std::intrinsics::prefetch_write_instruction(::std::ptr::dangling::<u8>(), 2);
        ::std::intrinsics::prefetch_write_instruction(X.as_ptr(), 3);
    }
}
