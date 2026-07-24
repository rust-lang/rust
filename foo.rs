#![feature(allocator_api)]

pub struct Drain<
    'a,
    T: 'a,
    A: std::alloc::Allocator + 'a = std::alloc::Global,
> {
    /// Index of tail to preserve
    pub tail_start: usize,
    /// Length of tail
    pub tail_len: usize,
    /// Current remaining range to remove
    pub iter: core::slice::Iter<'a, T>,
    pub vec: core::ptr::NonNull<Vec<T, A>>,
}

struct DropGuard<'r, 'a, T, A: std::alloc::Allocator>(&'r mut Drain<'a, T, A>);

fn main() {}