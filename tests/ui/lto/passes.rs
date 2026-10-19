//@ build-pass
//@ ignore-backends: gcc
//@ compile-flags: -O -Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=3 -C lto
#![crate_type = "lib"]

#[unsafe(no_mangle)]
pub extern "C" fn test(x: i32) -> i32 {
    if x > 3 { x * 7 } else { x - 1 }
}
