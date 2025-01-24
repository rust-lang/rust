//@ known-bug: #110395

const A: [(); 5] = [(), (), (), (), ()];

// Since the indexing is on a ZST, the addresses are all fine,
// but we should still catch the bad range.
const B: &[()] = unsafe { A.get_unchecked(3..1) };

fn main() {}
