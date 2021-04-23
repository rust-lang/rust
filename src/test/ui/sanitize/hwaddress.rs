// needs-sanitizer-support
// needs-sanitizer-hwaddress
//
// FIXME(#83989): codegen-units=1 triggers linker errors on aarch64-gnu
// compile-flags: -Z sanitizer=hwaddress -O -g -C codegen-units=16
//
// run-fail
// error-pattern: HWAddressSanitizer: tag-mismatch

#![feature(test)]

use std::hint::black_box;

fn main() {
    let xs = vec![0, 1, 2, 3];
    // Avoid optimizing everything out.
    let xs = black_box(xs.as_ptr());
    let code = unsafe { *xs.offset(4) };
    std::process::exit(code);
}
