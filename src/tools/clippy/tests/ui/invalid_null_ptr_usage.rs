fn main() {
    unsafe {
        let _slice: &[usize] = std::slice::from_raw_parts(std::ptr::null(), 0);
        let _slice: &[usize] = std::slice::from_raw_parts(std::ptr::null_mut(), 0);

        let _slice: &[usize] = std::slice::from_raw_parts_mut(std::ptr::null_mut(), 0);

        std::ptr::copy::<usize>(std::ptr::null(), std::ptr::NonNull::dangling().as_ptr(), 0);
        std::ptr::copy::<usize>(std::ptr::NonNull::dangling().as_ptr(), std::ptr::null_mut(), 0);

        std::ptr::copy_nonoverlapping::<usize>(std::ptr::null(), std::ptr::NonNull::dangling().as_ptr(), 0);
        std::ptr::copy_nonoverlapping::<usize>(std::ptr::NonNull::dangling().as_ptr(), std::ptr::null_mut(), 0);

        struct A; // zero sized struct
        assert_eq!(std::mem::size_of::<A>(), 0);

        let _a: A = std::ptr::read(std::ptr::null());
        let _a: A = std::ptr::read(std::ptr::null_mut());

        let _a: A = std::ptr::read_unaligned(std::ptr::null());
        let _a: A = std::ptr::read_unaligned(std::ptr::null_mut());

        let _a: A = std::ptr::read_volatile(std::ptr::null());
        let _a: A = std::ptr::read_volatile(std::ptr::null_mut());

        let _a: A = std::ptr::replace(std::ptr::null_mut(), A);
        let _slice: *const [usize] = std::ptr::slice_from_raw_parts(std::ptr::null_mut(), 0); // shouldn't lint
        let _slice: *const [usize] = std::ptr::slice_from_raw_parts_mut(std::ptr::null_mut(), 0);

        std::ptr::swap::<A>(std::ptr::null_mut(), &mut A);
        std::ptr::swap::<A>(&mut A, std::ptr::null_mut());

        std::ptr::swap_nonoverlapping::<A>(std::ptr::null_mut(), &mut A, 0);
        std::ptr::swap_nonoverlapping::<A>(&mut A, std::ptr::null_mut(), 0);

        std::ptr::write(std::ptr::null_mut(), A);

        std::ptr::write_unaligned(std::ptr::null_mut(), A);

        std::ptr::write_volatile(std::ptr::null_mut(), A);

        std::ptr::write_bytes::<usize>(std::ptr::null_mut(), 42, 0);
    }
}
