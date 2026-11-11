// Compiler:
//
// Run-time:
//   status: 0

// 128-bit atomics must be inlined lock-free wherever LLVM inlines them: a backend that lowers
// them to libatomic calls fails the link with `undefined reference to __atomic_load_16`.
// x86-64 needs: CARGO_TEST_FLAGS="-Ctarget-feature=+cmpxchg16b" ./y.sh test --cargo-tests -- atomic_u128

#![feature(core_intrinsics)]
#![allow(internal_features, unused_features)]

#[cfg(any(target_arch = "aarch64", target_feature = "cmpxchg16b"))]
mod atomics {
    use std::cell::UnsafeCell;
    use std::intrinsics::AtomicOrdering::*;
    use std::intrinsics::*;

    #[repr(align(16))]
    struct Cell<T>(UnsafeCell<T>);

    unsafe impl<T> Sync for Cell<T> {}

    const BIG: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;

    #[inline(never)]
    fn unsigned_operations(pointer: *mut u128) {
        unsafe {
            atomic_store::<u128, { SeqCst }, false>(pointer, BIG);
            assert_eq!(atomic_load::<u128, { Acquire }, false>(pointer), BIG);
            assert_eq!(atomic_load::<u128, { Relaxed }, false>(pointer), BIG);
            atomic_store::<u128, { Release }, false>(pointer, 5);
            assert_eq!(atomic_load::<u128, { SeqCst }, false>(pointer), 5);

            assert_eq!(atomic_xchg::<u128, { AcqRel }>(pointer, BIG), 5);
            assert_eq!(atomic_xadd::<u128, u128, { SeqCst }>(pointer, 1 << 64), BIG);
            assert_eq!(*pointer, BIG.wrapping_add(1 << 64));
            assert_eq!(
                atomic_xsub::<u128, u128, { Relaxed }>(pointer, 1 << 64),
                BIG.wrapping_add(1 << 64)
            );
            assert_eq!(atomic_and::<u128, u128, { Acquire }>(pointer, !0 << 64), BIG);
            assert_eq!(atomic_or::<u128, u128, { Release }>(pointer, 0xff), BIG & (!0 << 64));
            assert_eq!(
                atomic_xor::<u128, u128, { SeqCst }>(pointer, !0),
                (BIG & (!0 << 64)) | 0xff
            );
            assert_eq!(*pointer, !((BIG & (!0 << 64)) | 0xff));
            *pointer = 0b1100;
            assert_eq!(atomic_nand::<u128, u128, { SeqCst }>(pointer, 0b1010), 0b1100);
            assert_eq!(*pointer, !0b1000);

            *pointer = 10;
            assert_eq!(atomic_umax::<u128, { SeqCst }>(pointer, u128::MAX), 10);
            assert_eq!(atomic_umax::<u128, { SeqCst }>(pointer, 3), u128::MAX);
            assert_eq!(*pointer, u128::MAX);
            assert_eq!(atomic_umin::<u128, { Acquire }>(pointer, 7), u128::MAX);
            assert_eq!(atomic_umin::<u128, { Acquire }>(pointer, 9), 7);
            assert_eq!(*pointer, 7);

            assert_eq!(atomic_cxchg::<u128, { SeqCst }, { SeqCst }>(pointer, 7, BIG), (7, true));
            assert_eq!(atomic_cxchg::<u128, { Acquire }, { Relaxed }>(pointer, 7, 1), (BIG, false));
        }
    }

    #[inline(never)]
    fn signed_operations(pointer: *mut i128) {
        unsafe {
            *pointer = -5;
            assert_eq!(atomic_max::<i128, { SeqCst }>(pointer, 3), -5);
            assert_eq!(atomic_max::<i128, { SeqCst }>(pointer, i128::MIN), 3);
            assert_eq!(*pointer, 3);
            assert_eq!(atomic_min::<i128, { AcqRel }>(pointer, i128::MIN), 3);
            assert_eq!(atomic_min::<i128, { AcqRel }>(pointer, 0), i128::MIN);
            assert_eq!(*pointer, i128::MIN);
            assert_eq!(atomic_xadd::<i128, i128, { SeqCst }>(pointer, -1), i128::MIN);
            assert_eq!(atomic_load::<i128, { SeqCst }, false>(pointer), i128::MAX);
        }
    }

    static COUNTER: Cell<u128> = Cell(UnsafeCell::new(0));
    static MAXIMUM: Cell<u128> = Cell(UnsafeCell::new(0));

    // Each thread updates both halves, so a torn read-modify-write loses increments.
    fn concurrent_operations() {
        const THREADS: u128 = 8;
        const ITERATIONS: u128 = 10_000;
        let threads: Vec<_> = (0..THREADS)
            .map(|thread| {
                std::thread::spawn(move || unsafe {
                    for iteration in 0..ITERATIONS {
                        atomic_xadd::<u128, u128, { SeqCst }>(COUNTER.0.get(), (1 << 64) | 1);
                        atomic_umax::<u128, { SeqCst }>(
                            MAXIMUM.0.get(),
                            (iteration << 64) | thread,
                        );
                        let mut current = atomic_load::<u128, { Relaxed }, false>(COUNTER.0.get());
                        loop {
                            let (previous, success) =
                                atomic_cxchgweak::<u128, { SeqCst }, { Relaxed }>(
                                    COUNTER.0.get(),
                                    current,
                                    current.wrapping_add(1 << 32),
                                );
                            if success {
                                break;
                            }
                            current = previous;
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let total = THREADS * ITERATIONS;
        let counter = unsafe { atomic_load::<u128, { SeqCst }, false>(COUNTER.0.get()) };
        assert_eq!(counter, (total << 64) | (total << 32) | total);
        let maximum = unsafe { atomic_load::<u128, { SeqCst }, false>(MAXIMUM.0.get()) };
        assert_eq!(maximum >> 64, ITERATIONS - 1);
    }

    pub fn run() {
        let mut unsigned = Cell(UnsafeCell::new(0u128));
        unsigned_operations(unsigned.0.get_mut());
        let mut signed = Cell(UnsafeCell::new(0i128));
        signed_operations(signed.0.get_mut());
        concurrent_operations();
    }
}

fn main() {
    #[cfg(any(target_arch = "aarch64", target_feature = "cmpxchg16b"))]
    atomics::run();
}
