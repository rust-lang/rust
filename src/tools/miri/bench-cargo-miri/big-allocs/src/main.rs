//! This is a regression test for https://github.com/rust-lang/miri/issues/3637.
//! `Allocation`s are backed by a `Box<[u8]>`, which we create using `alloc_zeroed`, which should
//! make very large allocations cheap. But then we also need to not clone those `Allocation`s, or
//! we end up slow anyway.

fn main() {
    // We can't use too big of an allocation or this code will encounter an allocation failure in
    // CI. Since the allocation can't be huge, we need to do a few iterations so that the effect
    // we're trying to measure is clearly visible above the interpreter's startup time.
    // FIXME (https://github.com/rust-lang/miri/issues/4253): On 32bit targets, we can run out of
    // usable addresses if we don't reuse, leading to random test failures.
    let count = if cfg!(target_pointer_width = "32") { 8 } else { 12 };
    for _ in 0..count {
        drop(Vec::<u8>::with_capacity(512 * 1024 * 1024));
    }
}
