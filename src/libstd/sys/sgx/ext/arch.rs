//! SGX-specific access to architectural features.
//!
//! The functionality in this module is further documented in the Intel
//! Software Developer's Manual, Volume 3, Chapter 40.
#![unstable(feature = "sgx_platform", issue = "56975")]

use mem::MaybeUninit;

/// Wrapper struct to force 16-byte alignment.
#[repr(align(16))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align16<T>(pub T);

/// Wrapper struct to force 128-byte alignment.
#[repr(align(128))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align128<T>(pub T);

/// Wrapper struct to force 512-byte alignment.
#[repr(align(512))]
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Align512<T>(pub T);

const ENCLU_EREPORT: u32 = 0;
const ENCLU_EGETKEY: u32 = 1;

/// Call the `EGETKEY` instruction to obtain a 128-bit secret key.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn egetkey(request: &Align512<[u8; 512]>) -> Result<Align16<[u8; 16]>, u32> {
    unsafe {
        let mut out = MaybeUninit::uninitialized();
        let error;

        asm!(
            "enclu"
            : "={eax}"(error)
            : "{eax}"(ENCLU_EGETKEY),
              "{rbx}"(request),
              "{rcx}"(out.get_mut())
            : "flags"
        );

        match error {
            0 => Ok(out.into_inner()),
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
        let mut report = MaybeUninit::uninitialized();

        asm!(
            "enclu"
            : /* no output registers */
            : "{eax}"(ENCLU_EREPORT),
              "{rbx}"(targetinfo),
              "{rcx}"(reportdata),
              "{rdx}"(report.get_mut())
        );

        report.into_inner()
    }
}
