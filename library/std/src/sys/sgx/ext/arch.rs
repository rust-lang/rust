//! SGX-specific access to architectural features.
//!
//! The functionality in this module is further documented in the Intel
//! Software Developer's Manual, Volume 3, Chapter 40.
#![unstable(feature = "sgx_platform", issue = "56975")]

use crate::mem::MaybeUninit;
use core::slice;

/// Wrapper struct to force 16-byte alignment.
#[repr(align(16))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align16<T>(pub T);

/// Wrapper struct to force 64-byte alignment.
#[repr(align(64))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align64<T>(pub T);

/// Wrapper struct to force 128-byte alignment.
#[repr(align(128))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align128<T>(pub T);

/// Wrapper struct to force 512-byte alignment.
#[repr(align(512))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align512<T>(pub T);

use sgx_isa::Secinfo;
impl From<Secinfo> for Align64<[u8; 64]> {
    fn from(secinfo: Secinfo) -> Align64<[u8; 64]> {
        let mut arr = [0; 64];
        unsafe {
            arr.copy_from_slice(slice::from_raw_parts(&secinfo as *const Secinfo as *const _, 64))
        };
        Align64(arr)
    }
}

const ENCLU_EREPORT: u32 = 0;
const ENCLU_EGETKEY: u32 = 1;
const ENCLU_EACCEPT: u32 = 5;

/// Call the `EGETKEY` instruction to obtain a 128-bit secret key.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn egetkey(request: &Align512<[u8; 512]>) -> Result<Align16<[u8; 16]>, u32> {
    unsafe {
        let mut out = MaybeUninit::uninit();
        let error;

        asm!(
            "enclu",
            inlateout("eax") ENCLU_EGETKEY => error,
            in("rbx") request,
            in("rcx") out.as_mut_ptr(),
            options(nostack),
        );

        match error {
            0 => Ok(out.assume_init()),
            err => Err(err),
        }
    }
}

/// Call the `EREPORT` instruction.
///
/// This creates a cryptographic report describing the contents of the current
/// enclave. The report may be verified by the enclave described in
/// `targetinfo`.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn ereport(
    targetinfo: &Align512<[u8; 512]>,
    reportdata: &Align128<[u8; 64]>,
) -> Align512<[u8; 432]> {
    unsafe {
        let mut report = MaybeUninit::uninit();

        asm!(
            "enclu",
            in("eax") ENCLU_EREPORT,
            in("rbx") targetinfo,
            in("rcx") reportdata,
            in("rdx") report.as_mut_ptr(),
            options(preserves_flags, nostack),
        );

        report.assume_init()
    }
}

/// Call the `EACCEPT` instruction.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn eaccept(page: u64, secinfo: &Align64<[u8; 64]>) -> Result<(), u32> {
    let error: u32;
    unsafe {
        asm!(
            "enclu",
                 inlateout("rax") ENCLU_EACCEPT => error,
                 in("rbx") secinfo,
                 in("rcx") page,
                 // NOTE(#76738): ATT syntax is used to support LLVM 8 and 9.
                 options(att_syntax, nostack));
    }

    match error {
        0 => Ok(()),
        err => Err(err),
    }
}
