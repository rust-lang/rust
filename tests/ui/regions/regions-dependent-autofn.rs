//@ run-pass
// Test lifetimes are linked properly when we autoslice a vector.
// Issue #3148.


fn subslice<F>(v: F) -> F where F: FnOnce() { v }

fn both<F>(v: F) -> F where F: FnOnce() {
    subslice(subslice(v))
}

pub fn main() {
    both(main);
}
