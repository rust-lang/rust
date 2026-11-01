use std::mem::size_of;
use std::os::raw::{c_int, c_void};
use std::ptr;

unsafe fn swap_i32(lhs: *mut i32, rhs: *mut i32) {
    ptr::swap(lhs, rhs);
}

unsafe fn partition(
    arr: *mut i32,
    low: isize,
    high: isize,
    cmp: extern "C" fn(*const c_void, *const c_void) -> c_int,
) -> isize {
    let pivot = arr.offset(low);
    let mut i = low;
    let mut j = high;

    while i < j {
        while i <= high - 1 && cmp(arr.offset(i) as *const c_void, pivot as *const c_void) <= 0 {
            i += 1;
        }

        while j >= low + 1 && cmp(arr.offset(j) as *const c_void, pivot as *const c_void) > 0 {
            j -= 1;
        }

        if i < j {
            swap_i32(arr.offset(i), arr.offset(j));
        }
    }

    swap_i32(arr.offset(low), arr.offset(j));
    j
}

unsafe fn quicksort_rec(
    arr: *mut i32,
    low: isize,
    high: isize,
    cmp: extern "C" fn(*const c_void, *const c_void) -> c_int,
) {
    if low < high {
        let part = partition(arr, low, high, cmp);
        quicksort_rec(arr, low, part - 1, cmp);
        quicksort_rec(arr, part + 1, high, cmp);
    }
}

#[no_mangle]
pub extern "C" fn quickSort(
    base: *mut c_void,
    n: usize,
    size: usize,
    cmp: extern "C" fn(*const c_void, *const c_void) -> c_int,
) {
    if size != size_of::<i32>() {
        std::process::abort();
    }

    if n > 1 {
        unsafe {
            quicksort_rec(base as *mut i32, 0, (n as isize) - 1, cmp);
        }
    }
}
