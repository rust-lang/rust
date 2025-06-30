pub fn fill_bytes(bytes: &mut [u8]) {
    // Handle zero-byte request
    if bytes.is_empty() {
        return;
    }

    // Try EFI_RNG_PROTOCOL
    if rng_protocol::fill_bytes(bytes) {
        return;
    }

    // Fallback to rdrand if rng protocol missing.
    //
    // For real-world example, see [issue-13825](https://github.com/rust-lang/rust/issues/138252#issuecomment-2891270323)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if rdrand::fill_bytes(bytes) {
        return;
    }

    panic!("failed to generate random data");
}

mod rng_protocol {
    use r_efi::protocols::rng;

    use crate::sys::pal::helpers;

    pub(crate) fn fill_bytes(bytes: &mut [u8]) -> bool {
        if let Ok(handles) = helpers::locate_handles(rng::PROTOCOL_GUID) {
            for handle in handles {
                if let Ok(protocol) =
                    helpers::open_protocol::<rng::Protocol>(handle, rng::PROTOCOL_GUID)
                {
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
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// Port from [getrandom](https://github.com/rust-random/getrandom/blob/master/src/backends/rdrand.rs)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod rdrand {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use crate::arch::x86_64 as arch;
            use arch::_rdrand64_step as rdrand_step;
            type Word = u64;
        } else if #[cfg(target_arch = "x86")] {
            use crate::arch::x86 as arch;
            use arch::_rdrand32_step as rdrand_step;
            type Word = u32;
        }
    }

    static RDRAND_GOOD: crate::sync::LazyLock<bool> = crate::sync::LazyLock::new(is_rdrand_good);

    // Recommendation from "Intel® Digital Random Number Generator (DRNG) Software
    // Implementation Guide" - Section 5.2.1 and "Intel® 64 and IA-32 Architectures
    // Software Developer’s Manual" - Volume 1 - Section 7.3.17.1.
    const RETRY_LIMIT: usize = 10;

    unsafe fn rdrand() -> Option<Word> {
        for _ in 0..RETRY_LIMIT {
            let mut val = 0;
            if unsafe { rdrand_step(&mut val) } == 1 {
                return Some(val);
            }
        }
        None
    }

    // Run a small self-test to make sure we aren't repeating values
    // Adapted from Linux's test in arch/x86/kernel/cpu/rdrand.c
    // Fails with probability < 2^(-90) on 32-bit systems
    unsafe fn self_test() -> bool {
        // On AMD, RDRAND returns 0xFF...FF on failure, count it as a collision.
        let mut prev = Word::MAX;
        let mut fails = 0;
        for _ in 0..8 {
            match unsafe { rdrand() } {
                Some(val) if val == prev => fails += 1,
                Some(val) => prev = val,
                None => return false,
            };
        }
        fails <= 2
    }

    fn is_rdrand_good() -> bool {
        #[cfg(not(target_feature = "rdrand"))]
        {
            // SAFETY: All Rust x86 targets are new enough to have CPUID, and we
            // check that leaf 1 is supported before using it.
            let cpuid0 = unsafe { arch::__cpuid(0) };
            if cpuid0.eax < 1 {
                return false;
            }
            let cpuid1 = unsafe { arch::__cpuid(1) };

            let vendor_id =
                [cpuid0.ebx.to_le_bytes(), cpuid0.edx.to_le_bytes(), cpuid0.ecx.to_le_bytes()];
            if vendor_id == [*b"Auth", *b"enti", *b"cAMD"] {
                let mut family = (cpuid1.eax >> 8) & 0xF;
                if family == 0xF {
                    family += (cpuid1.eax >> 20) & 0xFF;
                }
                // AMD CPUs families before 17h (Zen) sometimes fail to set CF when
                // RDRAND fails after suspend. Don't use RDRAND on those families.
                // See https://bugzilla.redhat.com/show_bug.cgi?id=1150286
                if family < 0x17 {
                    return false;
                }
            }

            const RDRAND_FLAG: u32 = 1 << 30;
            if cpuid1.ecx & RDRAND_FLAG == 0 {
                return false;
            }
        }

        // SAFETY: We have already checked that rdrand is available.
        unsafe { self_test() }
    }

    unsafe fn rdrand_exact(dest: &mut [u8]) -> Option<()> {
        let mut chunks = dest.array_chunks_mut();
        for chunk in &mut chunks {
            *chunk = unsafe { rdrand() }?.to_ne_bytes();
        }

        let tail = chunks.into_remainder();
        let n = tail.len();
        if n > 0 {
            let src = unsafe { rdrand() }?.to_ne_bytes();
            tail.copy_from_slice(&src[..n]);
        }
        Some(())
    }

    pub(crate) fn fill_bytes(bytes: &mut [u8]) -> bool {
        if *RDRAND_GOOD { unsafe { rdrand_exact(bytes).is_some() } } else { false }
    }
}
