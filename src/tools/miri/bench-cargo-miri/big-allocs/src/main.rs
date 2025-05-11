//! This is a regression test for https://github.com/rust-lang/miri/issues/3637.
//! `Allocation`s are backed by a `Box<[u8]>`, which we create using `alloc_zeroed`, which should
//! make very large allocations cheap. But then we also need to not clone those `Allocation`s, or
//! we end up slow anyway.

fn main() {
    // We can't use too big of an allocation or this code will encounter an allocation failure in
    // CI. Since the allocation can't be huge, we need to do a few iterations so that the effect
    // we're trying to measure is clearly visible above the interpreter's startup time.
    for _ in 0..20 {
        drop(Vec::<u8>::with_capacity(512 * 1024 * 1024));
    }
}
