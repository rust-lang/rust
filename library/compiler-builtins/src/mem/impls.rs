use super::c_int;

#[inline(always)]
pub unsafe fn copy_forward(dest: *mut u8, src: *const u8, n: usize) {
    let mut i = 0;
    while i < n {
        *dest.offset(i as isize) = *src.offset(i as isize);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn copy_backward(dest: *mut u8, src: *const u8, n: usize) {
    // copy from end
    let mut i = n;
    while i != 0 {
        i -= 1;
        *dest.offset(i as isize) = *src.offset(i as isize);
    }
}

#[inline(always)]
pub unsafe fn set_bytes(s: *mut u8, c: u8, n: usize) {
    let mut i = 0;
    while i < n {
        *s.offset(i as isize) = c;
        i += 1;
    }
}
