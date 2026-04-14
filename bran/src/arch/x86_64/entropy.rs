//! x86_64 hardware entropy: RDRAND and RDSEED support.

/// Check if RDRAND is supported via CPUID leaf 1, ECX bit 30.
pub fn has_rdrand() -> bool {
    let ecx: u32;
    unsafe {
        // LLVM reserves rbx, so we use xchg to save/restore it
        core::arch::asm!(
            "xchg rbx, {tmp}",
            "mov eax, 1",
            "cpuid",
            "xchg rbx, {tmp}",
            tmp = out(reg) _,
            out("ecx") ecx,
            out("eax") _,
            out("edx") _,
            options(nostack, preserves_flags),
        );
    }
    (ecx >> 30) & 1 != 0
}

/// Check if RDSEED is supported via CPUID leaf 7, EBX bit 18.
pub fn has_rdseed() -> bool {
    let ebx_result: u64;
    unsafe {
        // LLVM reserves rbx, so we use xchg to swap it with a temp register.
        // After cpuid, rbx has the result, so we move it out then restore rbx.
        core::arch::asm!(
            "xchg rbx, {tmp}",   // save rbx into tmp
            "mov eax, 7",
            "xor ecx, ecx",
            "cpuid",             // result is in ebx
            "xchg rbx, {tmp}",   // swap: tmp now has cpuid result, rbx is restored
            tmp = out(reg) ebx_result,
            out("eax") _,
            out("ecx") _,
            out("edx") _,
            options(nostack, preserves_flags),
        );
    }
    ((ebx_result as u32) >> 18) & 1 != 0
}

/// Execute RDRAND and return the result. Retries up to `max_retries` times.
fn rdrand64(max_retries: u32) -> Option<u64> {
    for _ in 0..max_retries {
        let val: u64;
        let ok: u8;
        unsafe {
            core::arch::asm!(
                "rdrand {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = out(reg_byte) ok,
                options(nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
        // Small pause before retry
        core::hint::spin_loop();
    }
    None
}

/// Execute RDSEED and return the result. Retries up to `max_retries` times.
fn rdseed64(max_retries: u32) -> Option<u64> {
    for _ in 0..max_retries {
        let val: u64;
        let ok: u8;
        unsafe {
            core::arch::asm!(
                "rdseed {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = out(reg_byte) ok,
                options(nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
        core::hint::spin_loop();
    }
    None
}

/// Fill `dst` with hardware entropy bytes using RDSEED (preferred) or RDRAND.
/// Returns the number of bytes actually filled.
pub fn fill_entropy(dst: &mut [u8]) -> usize {
    let use_rdseed = has_rdseed();
    let use_rdrand = has_rdrand();

    if !use_rdrand && !use_rdseed {
        return 0;
    }

    let mut filled = 0;
    let mut prev: Option<u64> = None;
    let mut stuck_count: u32 = 0;
    const STUCK_THRESHOLD: u32 = 8;

    while filled < dst.len() {
        // Prefer RDSEED, fall back to RDRAND
        let val = if use_rdseed {
            rdseed64(10).or_else(|| if use_rdrand { rdrand64(10) } else { None })
        } else {
            rdrand64(10)
        };

        let val = match val {
            Some(v) => v,
            None => break, // HW RNG failure — stop filling
        };

        // Stuck-RNG health check
        if let Some(p) = prev {
            if p == val {
                stuck_count += 1;
                if stuck_count >= STUCK_THRESHOLD {
                    break; // RNG appears stuck
                }
            } else {
                stuck_count = 0;
            }
        }
        prev = Some(val);

        let bytes = val.to_ne_bytes();
        let chunk = (dst.len() - filled).min(8);
        dst[filled..filled + chunk].copy_from_slice(&bytes[..chunk]);
        filled += chunk;
    }

    filled
}
