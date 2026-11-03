//@ build-pass
//@ ignore-backends: gcc
//@ revisions: lto no_lto
//@ compile-flags: -O -Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=3
//@[lto] compile-flags: -C lto
//@[no_lto] compile-flags: -C codegen-units=1
#![crate_type = "lib"]

#[unsafe(no_mangle)]
pub extern "C" fn test(x: i32) -> i32 {
    if x > 3 { x * 7 } else { x - 1 }
}
