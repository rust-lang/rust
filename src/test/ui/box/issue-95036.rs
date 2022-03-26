// compile-flags: -O
// build-pass

#![feature(allocator_api, bench_black_box)]

pub fn main() {
    let mut node = Box::new_in([5u8], &std::alloc::Global);
    node[0] = 7u8;
    std::hint::black_box(node);
}
