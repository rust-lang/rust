//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//
//@ compile-flags: -Z sanitizer=address -O -g -C unsafe-allow-abi-mismatch=sanitizer
//
//@ run-fail-or-crash
//@ error-pattern: AddressSanitizer: stack-buffer-overflow
//@ error-pattern: 'xs' (line 14) <== Memory access at offset

use std::hint::black_box;

fn main() {
    let xs = [0, 1, 2, 3];
    // Avoid optimizing everything out.
    let xs = black_box(xs.as_ptr());
    let code = unsafe { *xs.offset(4) };
    std::process::exit(code);
}
