#![allow(dead_code)]

// pretty-expanded FIXME #23616

struct S<T> {
    a: T,
    b: usize,
}

fn range_<F>(lo: usize, hi: usize, mut it: F) where F: FnMut(usize) {
    let mut lo_ = lo;
    while lo_ < hi { it(lo_); lo_ += 1; }
}

fn create_index<T>(_index: Vec<S<T>> , _hash_fn: extern fn(T) -> usize) {
    range_(0, 256, |_i| {
        let _bucket: Vec<T> = Vec::new();
    })
}

pub fn main() { }
