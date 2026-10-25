// Regression test for issue 114275 `typeid::typeid_itanium_cxx_abi::transform_ty`
// was expecting array type lengths to be evaluated, this was causing an ICE.
//
//@ build-pass
//@ compile-flags: -Ccodegen-units=1 -Clto -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ needs-sanitizer-cfi

#![crate_type = "lib"]

#[repr(transparent)]
pub struct Array([u8; 1 * 1]);

pub extern "C" fn array() -> Array {
    loop {}
}
