//@ needs-sanitizer-support
//@ needs-sanitizer-hwaddress
//
//@ compile-flags: -Z sanitizer=hwaddress -O -g -C target-feature=+tagged-globals -C unsafe-allow-abi-mismatch=sanitizer
//
//@ run-fail
//@ error-pattern: HWAddressSanitizer: tag-mismatch

use std::hint::black_box;

fn main() {
    let xs = vec![0, 1, 2, 3];
    // Avoid optimizing everything out.
    let xs = black_box(xs.as_ptr());
    let code = unsafe { *xs.offset(4) };
    std::process::exit(code);
}

//~? WARN unknown and unstable feature specified for `-Ctarget-feature`: `tagged-globals`
