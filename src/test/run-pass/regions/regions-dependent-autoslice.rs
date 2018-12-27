// run-pass
// Test lifetimes are linked properly when we autoslice a vector.
// Issue #3148.

fn subslice1<'r>(v: &'r [usize]) -> &'r [usize] { v }

fn both<'r>(v: &'r [usize]) -> &'r [usize] {
    subslice1(subslice1(v))
}

pub fn main() {
    let v = vec![1,2,3];
    both(&v);
}
