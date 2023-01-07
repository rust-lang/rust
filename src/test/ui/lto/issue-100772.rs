// run-pass
// needs-sanitizer-cfi
// compile-flags: -Zsanitizer=cfi -Clto
// no-prefer-dynamic

#![feature(allocator_api)]

fn main() {
    let _ = Box::new_in(&[0, 1], &std::alloc::Global);
}
