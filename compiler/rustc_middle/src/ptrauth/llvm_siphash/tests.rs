use llvm_siphash_vectors::{EXPECTED64, TEST_KEY};

use super::*;

// These tests mirror the SipHash tests from LLVM's Support/SipHashTest.cpp.
// The implementation here is a faithful Rust translation of the LLVM 64-bit
// SipHash-2-4 implementation used for pointer authentication discriminators.
//
// TEST_KEY and EXPECTED64 are copied from the upstream SipHash reference
// vectors. Keeping these values identical ensures that the Rust implementation
// produces the same SipHash output as LLVM.
//
// Run with:
//   x.py test compiler/rustc_middle --test-args siphash

#[test]
fn siphash24_reference_vectors_64() {
    // Validate the SipHash-2-4 implementation against the upstream SipHash
    // reference vectors.
    //
    // Each input is the byte sequence [0, 1, ..., len-1] and is hashed using
    // the reference key TEST_KEY ([0, 1, ..., 15]). The expected values are
    // compared byte-for-byte with the LLVM reference implementation output
    // (EXPECTED64).
    //
    // This test verifies the correctness of the SipHash primitive itself,
    // independently of the pointer authentication discriminator logic.
    let mut input = [0u8; 64];

    for len in 0..EXPECTED64.len() {
        input[len] = len as u8;

        let hash = siphash_2_4_64_with_key(&input[..len], &TEST_KEY);

        assert_eq!(hash.to_le_bytes(), EXPECTED64[len]);
    }
}

#[test]
fn pointer_auth_stable_siphash() {
    // Validate the rustc pointer-authentication SipHash wrapper against the
    // values used by the LLVM implementation.
    //
    // The returned value is a 16-bit discriminator derived from the stable
    // SipHash-2-4 output.
    assert_eq!(llvm_pointer_auth_stable_siphash(b""), 0xE793);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"strlen"), 0xF468);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"_ZN1 ind; f"), 0x2D15);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"isa"), 0x6AE1);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"objc_class:superclass"), 0xB5AB,);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"block_descriptor"), 0xC0BB,);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"method_list_t"), 0xC310,);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"_Zptrkvttf"), 1);
    assert_eq!(llvm_pointer_auth_stable_siphash(b"_Zaflhllod"), 0xFFFF);
}
