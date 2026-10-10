use std::os::raw::{c_int, c_void};

#[link(name = "quicksort")]
extern "C" {
    fn quickSort(
        base: *mut c_void,
        n: usize,
        size: usize,
        cmp: extern "C" fn(*const c_void, *const c_void) -> c_int,
    );
}

extern "C" fn cmp_i32_ascending(a: *const c_void, b: *const c_void) -> c_int {
    unsafe {
        let x = *(a as *const i32);
        let y = *(b as *const i32);

        if x < y {
            -1
        } else if x > y {
            1
        } else {
            0
        }
    }
}

fn main() {
    let mut data: [i32; 5] = [4, 2, 5, 3, 1];
    println!("Before sorting: {:?}", data);

    unsafe {
        quickSort(
            data.as_mut_ptr() as *mut c_void,
            data.len(),
            std::mem::size_of::<i32>(),
            cmp_i32_ascending,
        );
    }

    println!("After sorting:  {:?}", data);
    assert!(data.windows(2).all(|w| w[0] <= w[1]));
}
