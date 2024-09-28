//@ only-x86_64
//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]
#![no_std]
#![feature(str_internals)]

extern crate alloc;

/// Ensure that the ascii-prefix loop for `str::to_lowercase` and `str::to_uppercase` uses vector
/// instructions.
///
/// The llvm ir should be the same for all targets that support some form of simd. Only targets
/// without any simd instructions would see scalarized ir.
/// Unfortunately, there is no `only-simd` directive to only run this test on only such platforms,
/// and using test revisions would still require the core libraries for all platforms.
// CHECK-LABEL: @lower_while_ascii
// CHECK: [[A:%[0-9]]] = load <16 x i8>
// CHECK-NEXT: [[B:%[0-9]]] = icmp slt <16 x i8> [[A]], zeroinitializer
// CHECK-NEXT: [[C:%[0-9]]] = bitcast <16 x i1> [[B]] to i16
#[no_mangle]
pub fn lower_while_ascii(s: &str) -> (alloc::string::String, &str) {
    alloc::str::convert_while_ascii(s, u8::to_ascii_lowercase)
}
