use fortanix_sgx_abi::Tcs;

/// Gets the ID for the current thread. The ID is guaranteed to be unique among
/// all currently running threads in the enclave, and it is guaranteed to be
/// constant for the lifetime of the thread. More specifically for SGX, there
/// is a one-to-one correspondence of the ID to the address of the TCS.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn current() -> Tcs {
    extern "C" {
        fn get_tcs_addr() -> *mut u8;
    }
    let addr = unsafe { get_tcs_addr() };
    match Tcs::new(addr) {
        Some(tcs) => tcs,
        None => rtabort!("TCS must not be placed at address zero (this is a linker error)"),
    }
}
