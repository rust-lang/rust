//@ build-pass
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Tsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer -Zunstable-options
//@ no-prefer-dynamic
//@ only-x86_64-unknown-linux-gnu
//@ ignore-backends: gcc

#![feature(allocator_api)]

fn main() {
    let _ = Box::new_in(&[0, 1], &std::alloc::Global);
}
