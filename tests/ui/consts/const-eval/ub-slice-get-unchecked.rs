#![feature(const_index, const_trait_impl)]

const A: [(); 5] = [(), (), (), (), ()];

// Since the indexing is on a ZST, the addresses are all fine,
// but we should still catch the bad range.
const B: &[()] = unsafe { A.get_unchecked(3..1) };
//~^ ERROR: slice::get_unchecked requires that the range is within the slice

fn main() {}
