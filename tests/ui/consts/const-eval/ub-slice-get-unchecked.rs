#![feature(const_slice_index)]

const A: [(); 5] = [(), (), (), (), ()];

// Since the indexing is on a ZST, the addresses are all fine,
// but we should still catch the bad range.
const B: &[()] = unsafe { A.get_unchecked(3..1) };

fn main() {}
