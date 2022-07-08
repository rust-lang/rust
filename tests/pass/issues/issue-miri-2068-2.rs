//@compile-flags: -Zmiri-disable-validation

use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let mut x = MaybeUninit::<i64>::uninit();
        // Put in a ptr.
        x.as_mut_ptr().cast::<&i32>().write_unaligned(&0);
        // Overwrite parts of that pointer with 'uninit' through a Scalar.
        let ptr = x.as_mut_ptr().cast::<i32>();
        *ptr = MaybeUninit::uninit().assume_init();
        // Reading this back should hence work fine.
        let _c = *ptr;
    }
}
