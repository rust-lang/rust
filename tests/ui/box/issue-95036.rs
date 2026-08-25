//@ compile-flags: -O
//@ build-pass

#![feature(allocator_api)]

#[inline(never)]
pub fn by_ref(node: &mut Box<[u8; 1], &std::alloc::Global>) {
    node[0] = 9u8;
}

pub fn main() {
    let mut node = Box::new_in([5u8], &std::alloc::Global);
    node[0] = 7u8;

    std::hint::black_box(node);

    let mut node = Box::new_in([5u8], &std::alloc::Global);

    by_ref(&mut node);

    std::hint::black_box(node);
}
