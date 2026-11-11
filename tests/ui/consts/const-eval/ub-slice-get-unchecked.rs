#![feature(const_index, const_trait_impl)]

const ZST_ARRAY: [(); 5] = [(), (), (), (), ()];

// Since the indexing is on a ZST, the addresses are all fine,
// but we should still catch the bad range.
const ZST_RANGE_OOB: &[()] = unsafe { ZST_ARRAY.get_unchecked(3..1) };
//~^ ERROR: slice::get_unchecked requires that the range is within the slice

const ZST_INDEX_OOB: &() = unsafe { ZST_ARRAY.get_unchecked(9) };
//~^ ERROR: slice::get_unchecked requires that the index is within the slice

const ARRAY: [i32; 5] = [1, 2, 3, 4, 5];

const INDEX_OOB: &i32 = unsafe { ARRAY.get_unchecked(9) };
//~^ ERROR: slice::get_unchecked requires that the index is within the slice

const INDEX_OOB_MUT: () = unsafe {
    let mut array = ARRAY;
    let _ = array.get_unchecked_mut(9);
    //~^ ERROR: slice::get_unchecked_mut requires that the index is within the slice
};

fn main() {}
