// run-pass

use std::mem::MaybeUninit;

// This test makes sure MU is still usable in consts
const X: () = {
    unsafe { MaybeUninit::<MaybeUninit<u8>>::uninit().assume_init(); }
    unsafe { MaybeUninit::<()>::uninit().assume_init(); }
};

fn main() {
    println!("{X:?}");
}
