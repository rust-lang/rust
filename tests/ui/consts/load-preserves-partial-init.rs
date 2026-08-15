//@ run-pass

// issue: https://github.com/rust-lang/rust/issues/69488
// Loads of partially-initialized data could produce completely-uninitialized results.
// Test to make sure that we no longer do such a "deinitializing" load.

// Or, equivalently: `MaybeUninit`.
pub union BagOfBits<T: Copy> {
    uninit: (),
    _storage: T,
}

pub const fn make_1u8_bag<T: Copy>() -> BagOfBits<T> {
    assert!(core::mem::size_of::<T>() >= 1);
    let mut bag = BagOfBits { uninit: () };
    unsafe { (&mut bag as *mut _ as *mut u8).write(1); };
    bag
}

pub fn check_bag<T: Copy>(bag: &BagOfBits<T>) {
    let val = unsafe { (bag as *const _ as *const u8).read() };
    assert_eq!(val, 1);
}

fn main() {
    check_bag(&make_1u8_bag::<[usize; 1]>()); // Fine
    check_bag(&make_1u8_bag::<usize>()); // Fine

    const CONST_ARRAY_BAG: BagOfBits<[usize; 1]> = make_1u8_bag();
    check_bag(&CONST_ARRAY_BAG); // Fine.
    const CONST_USIZE_BAG: BagOfBits<usize> = make_1u8_bag();

    // Used to panic since CTFE would make the entire `BagOfBits<usize>` uninit
    check_bag(&CONST_USIZE_BAG);
}
