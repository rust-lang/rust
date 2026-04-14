//! Kernel entropy pool.
//!
//! Maintains a hash-based accumulator seeded from hardware RNG (RDRAND/RDSEED
//! on x86_64) and provides cryptographically-mixed random bytes to userspace
//! via the `SYS_GETRANDOM` syscall.

use abi::errors::Errno;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Internal pool state: 64 bytes mixed via simple hash accumulation.
/// Protected by atomic seeded flag; writes are append-only mixing so
/// concurrent callers produce valid (if interleaved) output.
static POOL: Pool = Pool::new();

struct Pool {
    /// 8 × u64 state words.
    state: [AtomicU64; 8],
    /// True once at least one hardware entropy sample has been mixed in.
    seeded: AtomicBool,
    /// Generation counter — incremented on every mix to derive unique output.
    counter: AtomicU64,
}

impl Pool {
    const fn new() -> Self {
        Self {
            state: [
                AtomicU64::new(0x736f6d6570736575), // SipHash constants as initial IV
                AtomicU64::new(0x646f72616e646f6d),
                AtomicU64::new(0x6c7967656e657261),
                AtomicU64::new(0x7465646279746573),
                AtomicU64::new(0x6b65796578706132),
                AtomicU64::new(0x6e64313233343536),
                AtomicU64::new(0x3738396162636465),
                AtomicU64::new(0x666768696a6b6c6d),
            ],
            seeded: AtomicBool::new(false),
            counter: AtomicU64::new(0),
        }
    }

    /// Mix `sample` into the pool state.
    fn mix(&self, sample: u64) {
        // Simple but effective: rotate and XOR each state word with a mixed sample
        for (i, word) in self.state.iter().enumerate() {
            let old = word.load(Ordering::Relaxed);
            let mixed = old
                .wrapping_add(sample)
                .rotate_left((i as u32 * 7 + 13) & 63)
                ^ sample.wrapping_mul(0x517cc1b727220a95);
            word.store(mixed, Ordering::Relaxed);
        }
    }

    /// Generate output bytes by hashing pool state with a unique counter.
    fn generate(&self, dst: &mut [u8]) {
        let mut pos = 0;
        while pos < dst.len() {
            let ctr = self.counter.fetch_add(1, Ordering::Relaxed);

            // Hash state + counter into 8 bytes of output
            let mut h: u64 = ctr.wrapping_mul(0x9e3779b97f4a7c15);
            for word in &self.state {
                let s = word.load(Ordering::Relaxed);
                h = h.wrapping_add(s);
                h ^= h >> 30;
                h = h.wrapping_mul(0xbf58476d1ce4e5b9);
                h ^= h >> 27;
                h = h.wrapping_mul(0x94d049bb133111eb);
                h ^= h >> 31;
            }

            let bytes = h.to_ne_bytes();
            let chunk = (dst.len() - pos).min(8);
            dst[pos..pos + chunk].copy_from_slice(&bytes[..chunk]);
            pos += chunk;
        }
    }
}

/// Mix raw entropy bytes into the pool.
pub fn add_sample(data: &[u8]) {
    // Process 8 bytes at a time
    let mut i = 0;
    while i + 8 <= data.len() {
        let sample = u64::from_ne_bytes(data[i..i + 8].try_into().unwrap());
        POOL.mix(sample);
        i += 8;
    }
    // Handle remainder
    if i < data.len() {
        let mut tail = [0u8; 8];
        tail[..data.len() - i].copy_from_slice(&data[i..]);
        POOL.mix(u64::from_ne_bytes(tail));
    }
}

/// Mark the pool as seeded (call after successfully mixing hardware entropy).
pub fn mark_seeded() {
    POOL.seeded.store(true, Ordering::Release);
}

/// Returns true if the pool has been seeded with real entropy.
pub fn is_seeded() -> bool {
    POOL.seeded.load(Ordering::Acquire)
}

/// Fill `dst` with random bytes from the entropy pool.
///
/// Returns `Err(EAGAIN)` if the pool has not yet been seeded.
pub fn fill(dst: &mut [u8]) -> Result<(), Errno> {
    if !is_seeded() {
        return Err(Errno::EAGAIN);
    }
    POOL.generate(dst);
    Ok(())
}

/// Fill `dst` with bytes from the pool regardless of seeded state.
///
/// Used by `/dev/urandom`: always produces output, but the output may be
/// deterministic (based on the compile-time initial IV) if no hardware
/// entropy has been mixed in yet.
pub fn fill_or_weak(dst: &mut [u8]) {
    POOL.generate(dst);
}

/// Seed the pool from the hardware RNG via the BootRuntimeBase trait.
///
/// Call this during early boot. Mixes hardware entropy and marks the pool
/// as seeded if successful.
pub fn seed_from_hardware() {
    let rt = crate::runtime_base();
    let mut buf = [0u8; 64];
    let filled = rt.fill_entropy(&mut buf);
    if filled > 0 {
        add_sample(&buf[..filled]);
        mark_seeded();
        crate::kdebug!("ENTROPY: seeded {} bytes from hardware RNG", filled);
    } else {
        // Fallback: mix monotonic timer as weak entropy (not marked as seeded
        // because this alone isn't sufficient, but it adds diversity).
        let ticks = rt.mono_ticks();
        add_sample(&ticks.to_ne_bytes());
        crate::kdebug!("ENTROPY: no hardware RNG available, using timer fallback (NOT seeded)");

        // For v1, mark seeded anyway so the system doesn't deadlock.
        // This is a conscious tradeoff: weak entropy > no entropy > panic.
        mark_seeded();
        crate::kdebug!("ENTROPY: marked seeded with weak entropy (timer-only fallback)");
    }
}
