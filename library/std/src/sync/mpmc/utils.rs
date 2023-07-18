use crate::cell::Cell;
use crate::ops::{Deref, DerefMut};

/// Pads and aligns a value to the length of a cache line.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
// Starting from Intel's Sandy Bridge, spatial prefetcher is now pulling pairs of 64-byte cache
// lines at a time, so we have to align to 128 bytes rather than 64.
//
// Sources:
// - https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
// - https://github.com/facebook/folly/blob/1b5288e6eea6df074758f877c849b6e73bbb9fbb/folly/lang/Align.h#L107
//
// ARM's big.LITTLE architecture has asymmetric cores and "big" cores have 128-byte cache line size.
//
// Sources:
// - https://www.mono-project.com/news/2016/09/12/arm64-icache/
//
// powerpc64 has 128-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_ppc64x.go#L9
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "powerpc64",),
    repr(align(128))
)]
// arm, mips, mips64, and riscv64 have 32-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_arm.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mipsle.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips64x.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_riscv64.go#L7
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "riscv64",
    ),
    repr(align(32))
)]
// s390x has 256-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_s390x.go#L7
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
// x86 and wasm have 64-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/dda2991c2ea0c5914714469c4defc2562a907230/src/internal/cpu/cpu_x86.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_wasm.go#L7
//
// All others are assumed to have 64-byte cache line size.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "riscv64",
        target_arch = "s390x",
    )),
    repr(align(64))
)]
pub struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    /// Pads and aligns a value to the length of a cache line.
    pub fn new(value: T) -> CachePadded<T> {
        CachePadded::<T> { value }
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

const SPIN_LIMIT: u32 = 6;

/// Performs quadratic backoff in spin loops.
pub struct Backoff {
    step: Cell<u32>,
}

impl Backoff {
    /// Creates a new `Backoff`.
    pub fn new() -> Self {
        Backoff { step: Cell::new(0) }
    }

    /// Backs off using lightweight spinning.
    ///
    /// This method should be used for retrying an operation because another thread made
    /// progress. i.e. on CAS failure.
    #[inline]
    pub fn spin_light(&self) {
        let step = self.step.get().min(SPIN_LIMIT);
        for _ in 0..step.pow(2) {
            crate::hint::spin_loop();
        }

        self.step.set(self.step.get() + 1);
    }

    /// Backs off using heavyweight spinning.
    ///
    /// This method should be used in blocking loops where parking the thread is not an option.
    #[inline]
    pub fn spin_heavy(&self) {
        if self.step.get() <= SPIN_LIMIT {
            for _ in 0..self.step.get().pow(2) {
                crate::hint::spin_loop()
            }
        } else {
            crate::thread::yield_now();
        }

        self.step.set(self.step.get() + 1);
    }
}
