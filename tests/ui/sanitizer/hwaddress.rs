//@ needs-sanitizer-support
//@ needs-sanitizer-hwaddress
//
// FIXME(#83706): this test triggers errors on aarch64-gnu
//@ ignore-aarch64-unknown-linux-gnu
//
// FIXME(#83989): codegen-units=1 triggers linker errors on aarch64-gnu
//@ compile-flags: -Z sanitizer=hwaddress -O -g -C codegen-units=16 -C unsafe-allow-abi-mismatch=sanitizer
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
