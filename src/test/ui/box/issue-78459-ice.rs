// check-pass
#![feature(allocator_api)]

fn main() {
    Box::new_in((), &std::alloc::Global);
}
