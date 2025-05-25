use r_efi::protocols::rng;

use crate::sys::pal::helpers;

pub fn fill_bytes(bytes: &mut [u8]) {
    let handles =
        helpers::locate_handles(rng::PROTOCOL_GUID).expect("failed to generate random data");
    for handle in handles {
        if let Ok(protocol) = helpers::open_protocol::<rng::Protocol>(handle, rng::PROTOCOL_GUID) {
            let r = unsafe {
                ((*protocol.as_ptr()).get_rng)(
                    protocol.as_ptr(),
                    crate::ptr::null_mut(),
                    bytes.len(),
                    bytes.as_mut_ptr(),
                )
            };
            if r.is_error() {
                continue;
            } else {
                return;
            }
        }
    }

    // Fallback to rdrand if rng protocol missing.
    //
    // For real-world example, see [issue-13825](https://github.com/rust-lang/rust/issues/138252#issuecomment-2891270323)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if crate::is_x86_feature_detected!("rdrand") {
        #[cfg(target_arch = "x86_64")]
        for chunk in bytes.chunks_mut(core::mem::size_of::<u64>()) {
            let mut rand_val: u64 = 0;
            unsafe {
                if core::arch::x86_64::_rdrand64_step(&mut rand_val) == 0 {
                    panic!("failed to generate random data using rdrand");
                }
            }

            let bytes = rand_val.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }

        #[cfg(target_arch = "x86")]
        for chunk in bytes.chunks_mut(core::mem::size_of::<u32>()) {
            let mut rand_val: u32 = 0;
            unsafe {
                if core::arch::x86::_rdrand32_step(&mut rand_val) == 0 {
                    panic!("failed to generate random data using rdrand");
                }
            }

            let bytes = rand_val.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }

        return;
    }

    panic!("failed to generate random data");
}
