//! Compiler intrinsics.
//!
//! The corresponding definitions are in `compiler/rustc_codegen_llvm/src/intrinsic.rs`.
//! The corresponding const implementations are in `compiler/rustc_mir/src/interpret/intrinsics.rs`
//!
//! # Const intrinsics
//!
//! Note: any changes to the constness of intrinsics should be discussed with the language team.
//! This includes changes in the stability of the constness.
//!
//! In order to make an intrinsic usable at compile-time, one needs to copy the implementation
//! from <https://github.com/rust-lang/miri/blob/master/src/shims/intrinsics.rs> to
//! `compiler/rustc_mir/src/interpret/intrinsics.rs` and add a
//! `#[rustc_const_unstable(feature = "foo", issue = "01234")]` to the intrinsic.
//!
//! If an intrinsic is supposed to be used from a `const fn` with a `rustc_const_stable` attribute,
//! the intrinsic's attribute must be `rustc_const_stable`, too. Such a change should not be done
//! without T-lang consultation, because it bakes a feature into the language that cannot be
//! replicated in user code without compiler support.
//!
//! # Volatiles
//!
//! The volatile intrinsics provide operations intended to act on I/O
//! memory, which are guaranteed to not be reordered by the compiler
//! across other volatile intrinsics. See the LLVM documentation on
//! [[volatile]].
//!
//! [volatile]: https://llvm.org/docs/LangRef.html#volatile-memory-accesses
//!
//! # Atomics
//!
//! The atomic intrinsics provide common atomic operations on machine
//! words, with multiple possible memory orderings. They obey the same
//! semantics as C++11. See the LLVM documentation on [[atomics]].
//!
//! [atomics]: https://llvm.org/docs/Atomics.html
//!
//! A quick refresher on memory ordering:
//!
//! * Acquire - a barrier for acquiring a lock. Subsequent reads and writes
//!   take place after the barrier.
//! * Release - a barrier for releasing a lock. Preceding reads and writes
//!   take place before the barrier.
//! * Sequentially consistent - sequentially consistent operations are
//!   guaranteed to happen in order. This is the standard mode for working
//!   with atomic types and is equivalent to Java's `volatile`.

#![unstable(
    feature = "core_intrinsics",
    reason = "intrinsics are unlikely to ever be stabilized, instead \
                      they should be used through stabilized interfaces \
                      in the rest of the standard library",
    issue = "none"
)]
#![allow(missing_docs)]

use crate::marker::DiscriminantKind;
use crate::mem;

// These imports are used for simplifying intra-doc links
#[allow(unused_imports)]
#[cfg(all(target_has_atomic = "8", target_has_atomic = "32", target_has_atomic = "ptr"))]
use crate::sync::atomic::{self, AtomicBool, AtomicI32, AtomicIsize, AtomicU32, Ordering};

#[stable(feature = "drop_in_place", since = "1.8.0")]
#[rustc_deprecated(
    reason = "no longer an intrinsic - use `ptr::drop_in_place` directly",
    since = "1.52.0"
)]
#[inline]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // SAFETY: see `ptr::drop_in_place`
    unsafe { crate::ptr::drop_in_place(to_drop) }
}

extern "rust-intrinsic" {
    // N.B., these intrinsics take raw pointers because they mutate aliased
    // memory, which is not valid for either `&` or `&mut`.

    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::SeqCst`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::Acquire`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_acq<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::Release`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_rel<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::AcqRel`] as the `success` and [`Ordering::Acquire`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_acqrel<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::Relaxed`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::SeqCst`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::SeqCst`] as the `success` and [`Ordering::Acquire`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_failacq<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::Acquire`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_acq_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange` method by passing
    /// [`Ordering::AcqRel`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange`].
    pub fn atomic_cxchg_acqrel_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);

    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::SeqCst`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::Acquire`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_acq<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::Release`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_rel<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::AcqRel`] as the `success` and [`Ordering::Acquire`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_acqrel<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::Relaxed`] as both the `success` and `failure` parameters.
    /// For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::SeqCst`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::SeqCst`] as the `success` and [`Ordering::Acquire`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_failacq<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::Acquire`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_acq_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    /// Stores a value if the current value is the same as the `old` value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `compare_exchange_weak` method by passing
    /// [`Ordering::AcqRel`] as the `success` and [`Ordering::Relaxed`] as the
    /// `failure` parameters. For example, [`AtomicBool::compare_exchange_weak`].
    pub fn atomic_cxchgweak_acqrel_failrelaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);

    /// Loads the current value of the pointer.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `load` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::load`].
    pub fn atomic_load<T: Copy>(src: *const T) -> T;
    /// Loads the current value of the pointer.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `load` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::load`].
    pub fn atomic_load_acq<T: Copy>(src: *const T) -> T;
    /// Loads the current value of the pointer.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `load` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::load`].
    pub fn atomic_load_relaxed<T: Copy>(src: *const T) -> T;
    pub fn atomic_load_unordered<T: Copy>(src: *const T) -> T;

    /// Stores the value at the specified memory location.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `store` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::store`].
    pub fn atomic_store<T: Copy>(dst: *mut T, val: T);
    /// Stores the value at the specified memory location.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `store` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::store`].
    pub fn atomic_store_rel<T: Copy>(dst: *mut T, val: T);
    /// Stores the value at the specified memory location.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `store` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::store`].
    pub fn atomic_store_relaxed<T: Copy>(dst: *mut T, val: T);
    pub fn atomic_store_unordered<T: Copy>(dst: *mut T, val: T);

    /// Stores the value at the specified memory location, returning the old value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `swap` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::swap`].
    pub fn atomic_xchg<T: Copy>(dst: *mut T, src: T) -> T;
    /// Stores the value at the specified memory location, returning the old value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `swap` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::swap`].
    pub fn atomic_xchg_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Stores the value at the specified memory location, returning the old value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `swap` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::swap`].
    pub fn atomic_xchg_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Stores the value at the specified memory location, returning the old value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `swap` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::swap`].
    pub fn atomic_xchg_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Stores the value at the specified memory location, returning the old value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `swap` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::swap`].
    pub fn atomic_xchg_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Adds to the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_add` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicIsize::fetch_add`].
    pub fn atomic_xadd<T: Copy>(dst: *mut T, src: T) -> T;
    /// Adds to the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_add` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicIsize::fetch_add`].
    pub fn atomic_xadd_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Adds to the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_add` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicIsize::fetch_add`].
    pub fn atomic_xadd_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Adds to the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_add` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicIsize::fetch_add`].
    pub fn atomic_xadd_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Adds to the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_add` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicIsize::fetch_add`].
    pub fn atomic_xadd_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Subtract from the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_sub` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
    pub fn atomic_xsub<T: Copy>(dst: *mut T, src: T) -> T;
    /// Subtract from the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_sub` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
    pub fn atomic_xsub_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Subtract from the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_sub` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
    pub fn atomic_xsub_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Subtract from the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_sub` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
    pub fn atomic_xsub_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Subtract from the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_sub` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
    pub fn atomic_xsub_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Bitwise and with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_and` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_and`].
    pub fn atomic_and<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise and with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_and` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_and`].
    pub fn atomic_and_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise and with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_and` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_and`].
    pub fn atomic_and_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise and with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_and` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_and`].
    pub fn atomic_and_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise and with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_and` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_and`].
    pub fn atomic_and_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Bitwise nand with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`AtomicBool`] type via the `fetch_nand` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_nand`].
    pub fn atomic_nand<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise nand with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`AtomicBool`] type via the `fetch_nand` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_nand`].
    pub fn atomic_nand_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise nand with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`AtomicBool`] type via the `fetch_nand` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_nand`].
    pub fn atomic_nand_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise nand with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`AtomicBool`] type via the `fetch_nand` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_nand`].
    pub fn atomic_nand_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise nand with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`AtomicBool`] type via the `fetch_nand` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_nand`].
    pub fn atomic_nand_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Bitwise or with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_or` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_or`].
    pub fn atomic_or<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise or with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_or` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_or`].
    pub fn atomic_or_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise or with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_or` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_or`].
    pub fn atomic_or_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise or with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_or` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_or`].
    pub fn atomic_or_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise or with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_or` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_or`].
    pub fn atomic_or_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Bitwise xor with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_xor` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_xor`].
    pub fn atomic_xor<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise xor with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_xor` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_xor`].
    pub fn atomic_xor_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise xor with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_xor` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_xor`].
    pub fn atomic_xor_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise xor with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_xor` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_xor`].
    pub fn atomic_xor_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Bitwise xor with the current value, returning the previous value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] types via the `fetch_xor` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_xor`].
    pub fn atomic_xor_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Maximum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_max` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicI32::fetch_max`].
    pub fn atomic_max<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_max` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicI32::fetch_max`].
    pub fn atomic_max_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_max` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicI32::fetch_max`].
    pub fn atomic_max_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_max` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicI32::fetch_max`].
    pub fn atomic_max_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_max` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicI32::fetch_max`].
    pub fn atomic_max_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Minimum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_min` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicI32::fetch_min`].
    pub fn atomic_min<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_min` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicI32::fetch_min`].
    pub fn atomic_min_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_min` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicI32::fetch_min`].
    pub fn atomic_min_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_min` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicI32::fetch_min`].
    pub fn atomic_min_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using a signed comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] signed integer types via the `fetch_min` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicI32::fetch_min`].
    pub fn atomic_min_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Minimum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_min` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicU32::fetch_min`].
    pub fn atomic_umin<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_min` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicU32::fetch_min`].
    pub fn atomic_umin_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_min` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicU32::fetch_min`].
    pub fn atomic_umin_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_min` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicU32::fetch_min`].
    pub fn atomic_umin_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Minimum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_min` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicU32::fetch_min`].
    pub fn atomic_umin_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// Maximum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_max` method by passing
    /// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicU32::fetch_max`].
    pub fn atomic_umax<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_max` method by passing
    /// [`Ordering::Acquire`] as the `order`. For example, [`AtomicU32::fetch_max`].
    pub fn atomic_umax_acq<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_max` method by passing
    /// [`Ordering::Release`] as the `order`. For example, [`AtomicU32::fetch_max`].
    pub fn atomic_umax_rel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_max` method by passing
    /// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicU32::fetch_max`].
    pub fn atomic_umax_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
    /// Maximum with the current value using an unsigned comparison.
    ///
    /// The stabilized version of this intrinsic is available on the
    /// [`atomic`] unsigned integer types via the `fetch_max` method by passing
    /// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicU32::fetch_max`].
    pub fn atomic_umax_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

    /// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
    /// if supported; otherwise, it is a no-op.
    /// Prefetches have no effect on the behavior of the program but can change its performance
    /// characteristics.
    ///
    /// The `locality` argument must be a constant integer and is a temporal locality specifier
    /// ranging from (0) - no locality, to (3) - extremely local keep in cache.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn prefetch_read_data<T>(data: *const T, locality: i32);
    /// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
    /// if supported; otherwise, it is a no-op.
    /// Prefetches have no effect on the behavior of the program but can change its performance
    /// characteristics.
    ///
    /// The `locality` argument must be a constant integer and is a temporal locality specifier
    /// ranging from (0) - no locality, to (3) - extremely local keep in cache.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn prefetch_write_data<T>(data: *const T, locality: i32);
    /// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
    /// if supported; otherwise, it is a no-op.
    /// Prefetches have no effect on the behavior of the program but can change its performance
    /// characteristics.
    ///
    /// The `locality` argument must be a constant integer and is a temporal locality specifier
    /// ranging from (0) - no locality, to (3) - extremely local keep in cache.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn prefetch_read_instruction<T>(data: *const T, locality: i32);
    /// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
    /// if supported; otherwise, it is a no-op.
    /// Prefetches have no effect on the behavior of the program but can change its performance
    /// characteristics.
    ///
    /// The `locality` argument must be a constant integer and is a temporal locality specifier
    /// ranging from (0) - no locality, to (3) - extremely local keep in cache.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn prefetch_write_instruction<T>(data: *const T, locality: i32);
}

extern "rust-intrinsic" {
    /// An atomic fence.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::fence`] by passing [`Ordering::SeqCst`]
    /// as the `order`.
    pub fn atomic_fence();
    /// An atomic fence.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::fence`] by passing [`Ordering::Acquire`]
    /// as the `order`.
    pub fn atomic_fence_acq();
    /// An atomic fence.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::fence`] by passing [`Ordering::Release`]
    /// as the `order`.
    pub fn atomic_fence_rel();
    /// An atomic fence.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::fence`] by passing [`Ordering::AcqRel`]
    /// as the `order`.
    pub fn atomic_fence_acqrel();

    /// A compiler-only memory barrier.
    ///
    /// Memory accesses will never be reordered across this barrier by the
    /// compiler, but no instructions will be emitted for it. This is
    /// appropriate for operations on the same thread that may be preempted,
    /// such as when interacting with signal handlers.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::compiler_fence`] by passing [`Ordering::SeqCst`]
    /// as the `order`.
    pub fn atomic_singlethreadfence();
    /// A compiler-only memory barrier.
    ///
    /// Memory accesses will never be reordered across this barrier by the
    /// compiler, but no instructions will be emitted for it. This is
    /// appropriate for operations on the same thread that may be preempted,
    /// such as when interacting with signal handlers.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::compiler_fence`] by passing [`Ordering::Acquire`]
    /// as the `order`.
    pub fn atomic_singlethreadfence_acq();
    /// A compiler-only memory barrier.
    ///
    /// Memory accesses will never be reordered across this barrier by the
    /// compiler, but no instructions will be emitted for it. This is
    /// appropriate for operations on the same thread that may be preempted,
    /// such as when interacting with signal handlers.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::compiler_fence`] by passing [`Ordering::Release`]
    /// as the `order`.
    pub fn atomic_singlethreadfence_rel();
    /// A compiler-only memory barrier.
    ///
    /// Memory accesses will never be reordered across this barrier by the
    /// compiler, but no instructions will be emitted for it. This is
    /// appropriate for operations on the same thread that may be preempted,
    /// such as when interacting with signal handlers.
    ///
    /// The stabilized version of this intrinsic is available in
    /// [`atomic::compiler_fence`] by passing [`Ordering::AcqRel`]
    /// as the `order`.
    pub fn atomic_singlethreadfence_acqrel();

    /// Magic intrinsic that derives its meaning from attributes
    /// attached to the function.
    ///
    /// For example, dataflow uses this to inject static assertions so
    /// that `rustc_peek(potentially_uninitialized)` would actually
    /// double-check that dataflow did indeed compute that it is
    /// uninitialized at that point in the control flow.
    ///
    /// This intrinsic should not be used outside of the compiler.
    pub fn rustc_peek<T>(_: T) -> T;

    /// Aborts the execution of the process.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// [`std::process::abort`](../../std/process/fn.abort.html) is to be preferred if possible,
    /// as its behaviour is more user-friendly and more stable.
    ///
    /// The current implementation of `intrinsics::abort` (ab)uses a debug trap.  On Unix, the
    /// process will probably die of a signal like `SIGABRT`, `SIGILL`, `SIGTRAP`, `SIGSEGV` or
    /// `SIGBUS`.  The precise behaviour is not guaranteed and not stable.
    pub fn abort() -> !;

    /// Informs the optimizer that this point in the code is not reachable,
    /// enabling further optimizations.
    ///
    /// N.B., this is very different from the `unreachable!()` macro: Unlike the
    /// macro, which panics when it is executed, it is *undefined behavior* to
    /// reach code marked with this function.
    ///
    /// The stabilized version of this intrinsic is [`core::hint::unreachable_unchecked`].
    #[rustc_const_unstable(feature = "const_unreachable_unchecked", issue = "53188")]
    pub fn unreachable() -> !;

    /// Informs the optimizer that a condition is always true.
    /// If the condition is false, the behavior is undefined.
    ///
    /// No code is generated for this intrinsic, but the optimizer will try
    /// to preserve it (and its condition) between passes, which may interfere
    /// with optimization of surrounding code and reduce performance. It should
    /// not be used if the invariant can be discovered by the optimizer on its
    /// own, or if it does not enable any significant optimizations.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_assume", issue = "76972")]
    pub fn assume(b: bool);

    /// Hints to the compiler that branch condition is likely to be true.
    /// Returns the value passed to it.
    ///
    /// Any use other than with `if` statements will probably not have an effect.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_likely", issue = "none")]
    pub fn likely(b: bool) -> bool;

    /// Hints to the compiler that branch condition is likely to be false.
    /// Returns the value passed to it.
    ///
    /// Any use other than with `if` statements will probably not have an effect.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_likely", issue = "none")]
    pub fn unlikely(b: bool) -> bool;

    /// Executes a breakpoint trap, for inspection by a debugger.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn breakpoint();

    /// The size of a type in bytes.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// More specifically, this is the offset in bytes between successive
    /// items of the same type, including alignment padding.
    ///
    /// The stabilized version of this intrinsic is [`core::mem::size_of`].
    #[rustc_const_stable(feature = "const_size_of", since = "1.40.0")]
    pub fn size_of<T>() -> usize;

    /// The minimum alignment of a type.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is [`core::mem::align_of`].
    #[rustc_const_stable(feature = "const_min_align_of", since = "1.40.0")]
    pub fn min_align_of<T>() -> usize;
    /// The preferred alignment of a type.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_pref_align_of", issue = "none")]
    pub fn pref_align_of<T>() -> usize;

    /// The size of the referenced value in bytes.
    ///
    /// The stabilized version of this intrinsic is [`mem::size_of_val`].
    #[rustc_const_unstable(feature = "const_size_of_val", issue = "46571")]
    pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;
    /// The required alignment of the referenced value.
    ///
    /// The stabilized version of this intrinsic is [`core::mem::align_of_val`].
    #[rustc_const_unstable(feature = "const_align_of_val", issue = "46571")]
    pub fn min_align_of_val<T: ?Sized>(_: *const T) -> usize;

    /// Gets a static string slice containing the name of a type.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is [`core::any::type_name`].
    #[rustc_const_unstable(feature = "const_type_name", issue = "63084")]
    pub fn type_name<T: ?Sized>() -> &'static str;

    /// Gets an identifier which is globally unique to the specified type. This
    /// function will return the same value for a type regardless of whichever
    /// crate it is invoked in.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is [`core::any::TypeId::of`].
    #[rustc_const_unstable(feature = "const_type_id", issue = "77125")]
    pub fn type_id<T: ?Sized + 'static>() -> u64;

    /// A guard for unsafe functions that cannot ever be executed if `T` is uninhabited:
    /// This will statically either panic, or do nothing.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_assert_type", issue = "none")]
    pub fn assert_inhabited<T>();

    /// A guard for unsafe functions that cannot ever be executed if `T` does not permit
    /// zero-initialization: This will statically either panic, or do nothing.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn assert_zero_valid<T>();

    /// A guard for unsafe functions that cannot ever be executed if `T` has invalid
    /// bit patterns: This will statically either panic, or do nothing.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn assert_uninit_valid<T>();

    /// Gets a reference to a static `Location` indicating where it was called.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// Consider using [`core::panic::Location::caller`] instead.
    #[rustc_const_unstable(feature = "const_caller_location", issue = "76156")]
    pub fn caller_location() -> &'static crate::panic::Location<'static>;

    /// Moves a value out of scope without running drop glue.
    ///
    /// This exists solely for [`mem::forget_unsized`]; normal `forget` uses
    /// `ManuallyDrop` instead.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    #[rustc_const_unstable(feature = "const_intrinsic_forget", issue = "none")]
    pub fn forget<T: ?Sized>(_: T);

    /// Reinterprets the bits of a value of one type as another type.
    ///
    /// Both types must have the same size. Neither the original, nor the result,
    /// may be an [invalid value](../../nomicon/what-unsafe-does.html).
    ///
    /// `transmute` is semantically equivalent to a bitwise move of one type
    /// into another. It copies the bits from the source value into the
    /// destination value, then forgets the original. It's equivalent to C's
    /// `memcpy` under the hood, just like `transmute_copy`.
    ///
    /// Because `transmute` is a by-value operation, alignment of the *transmuted values
    /// themselves* is not a concern. As with any other function, the compiler already ensures
    /// both `T` and `U` are properly aligned. However, when transmuting values that *point
    /// elsewhere* (such as pointers, references, boxes…), the caller has to ensure proper
    /// alignment of the pointed-to values.
    ///
    /// `transmute` is **incredibly** unsafe. There are a vast number of ways to
    /// cause [undefined behavior][ub] with this function. `transmute` should be
    /// the absolute last resort.
    ///
    /// The [nomicon](../../nomicon/transmutes.html) has additional
    /// documentation.
    ///
    /// [ub]: ../../reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// There are a few things that `transmute` is really useful for.
    ///
    /// Turning a pointer into a function pointer. This is *not* portable to
    /// machines where function pointers and data pointers have different sizes.
    ///
    /// ```
    /// fn foo() -> i32 {
    ///     0
    /// }
    /// let pointer = foo as *const ();
    /// let function = unsafe {
    ///     std::mem::transmute::<*const (), fn() -> i32>(pointer)
    /// };
    /// assert_eq!(function(), 0);
    /// ```
    ///
    /// Extending a lifetime, or shortening an invariant lifetime. This is
    /// advanced, very unsafe Rust!
    ///
    /// ```
    /// struct R<'a>(&'a i32);
    /// unsafe fn extend_lifetime<'b>(r: R<'b>) -> R<'static> {
    ///     std::mem::transmute::<R<'b>, R<'static>>(r)
    /// }
    ///
    /// unsafe fn shorten_invariant_lifetime<'b, 'c>(r: &'b mut R<'static>)
    ///                                              -> &'b mut R<'c> {
    ///     std::mem::transmute::<&'b mut R<'static>, &'b mut R<'c>>(r)
    /// }
    /// ```
    ///
    /// # Alternatives
    ///
    /// Don't despair: many uses of `transmute` can be achieved through other means.
    /// Below are common applications of `transmute` which can be replaced with safer
    /// constructs.
    ///
    /// Turning raw bytes(`&[u8]`) to `u32`, `f64`, etc.:
    ///
    /// ```
    /// let raw_bytes = [0x78, 0x56, 0x34, 0x12];
    ///
    /// let num = unsafe {
    ///     std::mem::transmute::<[u8; 4], u32>(raw_bytes)
    /// };
    ///
    /// // use `u32::from_ne_bytes` instead
    /// let num = u32::from_ne_bytes(raw_bytes);
    /// // or use `u32::from_le_bytes` or `u32::from_be_bytes` to specify the endianness
    /// let num = u32::from_le_bytes(raw_bytes);
    /// assert_eq!(num, 0x12345678);
    /// let num = u32::from_be_bytes(raw_bytes);
    /// assert_eq!(num, 0x78563412);
    /// ```
    ///
    /// Turning a pointer into a `usize`:
    ///
    /// ```
    /// let ptr = &0;
    /// let ptr_num_transmute = unsafe {
    ///     std::mem::transmute::<&i32, usize>(ptr)
    /// };
    ///
    /// // Use an `as` cast instead
    /// let ptr_num_cast = ptr as *const i32 as usize;
    /// ```
    ///
    /// Turning a `*mut T` into an `&mut T`:
    ///
    /// ```
    /// let ptr: *mut i32 = &mut 0;
    /// let ref_transmuted = unsafe {
    ///     std::mem::transmute::<*mut i32, &mut i32>(ptr)
    /// };
    ///
    /// // Use a reborrow instead
    /// let ref_casted = unsafe { &mut *ptr };
    /// ```
    ///
    /// Turning an `&mut T` into an `&mut U`:
    ///
    /// ```
    /// let ptr = &mut 0;
    /// let val_transmuted = unsafe {
    ///     std::mem::transmute::<&mut i32, &mut u32>(ptr)
    /// };
    ///
    /// // Now, put together `as` and reborrowing - note the chaining of `as`
    /// // `as` is not transitive
    /// let val_casts = unsafe { &mut *(ptr as *mut i32 as *mut u32) };
    /// ```
    ///
    /// Turning an `&str` into an `&[u8]`:
    ///
    /// ```
    /// // this is not a good way to do this.
    /// let slice = unsafe { std::mem::transmute::<&str, &[u8]>("Rust") };
    /// assert_eq!(slice, &[82, 117, 115, 116]);
    ///
    /// // You could use `str::as_bytes`
    /// let slice = "Rust".as_bytes();
    /// assert_eq!(slice, &[82, 117, 115, 116]);
    ///
    /// // Or, just use a byte string, if you have control over the string
    /// // literal
    /// assert_eq!(b"Rust", &[82, 117, 115, 116]);
    /// ```
    ///
    /// Turning a `Vec<&T>` into a `Vec<Option<&T>>`.
    ///
    /// To transmute the inner type of the contents of a container, you must make sure to not
    /// violate any of the container's invariants. For `Vec`, this means that both the size
    /// *and alignment* of the inner types have to match. Other containers might rely on the
    /// size of the type, alignment, or even the `TypeId`, in which case transmuting wouldn't
    /// be possible at all without violating the container invariants.
    ///
    /// ```
    /// let store = [0, 1, 2, 3];
    /// let v_orig = store.iter().collect::<Vec<&i32>>();
    ///
    /// // clone the vector as we will reuse them later
    /// let v_clone = v_orig.clone();
    ///
    /// // Using transmute: this relies on the unspecified data layout of `Vec`, which is a
    /// // bad idea and could cause Undefined Behavior.
    /// // However, it is no-copy.
    /// let v_transmuted = unsafe {
    ///     std::mem::transmute::<Vec<&i32>, Vec<Option<&i32>>>(v_clone)
    /// };
    ///
    /// let v_clone = v_orig.clone();
    ///
    /// // This is the suggested, safe way.
    /// // It does copy the entire vector, though, into a new array.
    /// let v_collected = v_clone.into_iter()
    ///                          .map(Some)
    ///                          .collect::<Vec<Option<&i32>>>();
    ///
    /// let v_clone = v_orig.clone();
    ///
    /// // This is the proper no-copy, unsafe way of "transmuting" a `Vec`, without relying on the
    /// // data layout. Instead of literally calling `transmute`, we perform a pointer cast, but
    /// // in terms of converting the original inner type (`&i32`) to the new one (`Option<&i32>`),
    /// // this has all the same caveats. Besides the information provided above, also consult the
    /// // [`from_raw_parts`] documentation.
    /// let v_from_raw = unsafe {
    // FIXME Update this when vec_into_raw_parts is stabilized
    ///     // Ensure the original vector is not dropped.
    ///     let mut v_clone = std::mem::ManuallyDrop::new(v_clone);
    ///     Vec::from_raw_parts(v_clone.as_mut_ptr() as *mut Option<&i32>,
    ///                         v_clone.len(),
    ///                         v_clone.capacity())
    /// };
    /// ```
    ///
    /// [`from_raw_parts`]: ../../std/vec/struct.Vec.html#method.from_raw_parts
    ///
    /// Implementing `split_at_mut`:
    ///
    /// ```
    /// use std::{slice, mem};
    ///
    /// // There are multiple ways to do this, and there are multiple problems
    /// // with the following (transmute) way.
    /// fn split_at_mut_transmute<T>(slice: &mut [T], mid: usize)
    ///                              -> (&mut [T], &mut [T]) {
    ///     let len = slice.len();
    ///     assert!(mid <= len);
    ///     unsafe {
    ///         let slice2 = mem::transmute::<&mut [T], &mut [T]>(slice);
    ///         // first: transmute is not type safe; all it checks is that T and
    ///         // U are of the same size. Second, right here, you have two
    ///         // mutable references pointing to the same memory.
    ///         (&mut slice[0..mid], &mut slice2[mid..len])
    ///     }
    /// }
    ///
    /// // This gets rid of the type safety problems; `&mut *` will *only* give
    /// // you an `&mut T` from an `&mut T` or `*mut T`.
    /// fn split_at_mut_casts<T>(slice: &mut [T], mid: usize)
    ///                          -> (&mut [T], &mut [T]) {
    ///     let len = slice.len();
    ///     assert!(mid <= len);
    ///     unsafe {
    ///         let slice2 = &mut *(slice as *mut [T]);
    ///         // however, you still have two mutable references pointing to
    ///         // the same memory.
    ///         (&mut slice[0..mid], &mut slice2[mid..len])
    ///     }
    /// }
    ///
    /// // This is how the standard library does it. This is the best method, if
    /// // you need to do something like this
    /// fn split_at_stdlib<T>(slice: &mut [T], mid: usize)
    ///                       -> (&mut [T], &mut [T]) {
    ///     let len = slice.len();
    ///     assert!(mid <= len);
    ///     unsafe {
    ///         let ptr = slice.as_mut_ptr();
    ///         // This now has three mutable references pointing at the same
    ///         // memory. `slice`, the rvalue ret.0, and the rvalue ret.1.
    ///         // `slice` is never used after `let ptr = ...`, and so one can
    ///         // treat it as "dead", and therefore, you only have two real
    ///         // mutable slices.
    ///         (slice::from_raw_parts_mut(ptr, mid),
    ///          slice::from_raw_parts_mut(ptr.add(mid), len - mid))
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    // NOTE: While this makes the intrinsic const stable, we have some custom code in const fn
    // checks that prevent its use within `const fn`.
    #[rustc_const_stable(feature = "const_transmute", since = "1.46.0")]
    #[rustc_diagnostic_item = "transmute"]
    pub fn transmute<T, U>(e: T) -> U;

    /// Returns `true` if the actual type given as `T` requires drop
    /// glue; returns `false` if the actual type provided for `T`
    /// implements `Copy`.
    ///
    /// If the actual type neither requires drop glue nor implements
    /// `Copy`, then the return value of this function is unspecified.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is [`mem::needs_drop`](crate::mem::needs_drop).
    #[rustc_const_stable(feature = "const_needs_drop", since = "1.40.0")]
    pub fn needs_drop<T>() -> bool;

    /// Calculates the offset from a pointer.
    ///
    /// This is implemented as an intrinsic to avoid converting to and from an
    /// integer, since the conversion would throw away aliasing information.
    ///
    /// # Safety
    ///
    /// Both the starting and resulting pointer must be either in bounds or one
    /// byte past the end of an allocated object. If either pointer is out of
    /// bounds or arithmetic overflow occurs then any further use of the
    /// returned value will result in undefined behavior.
    ///
    /// The stabilized version of this intrinsic is [`pointer::offset`].
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[rustc_const_unstable(feature = "const_ptr_offset", issue = "71499")]
    pub fn offset<T>(dst: *const T, offset: isize) -> *const T;

    /// Calculates the offset from a pointer, potentially wrapping.
    ///
    /// This is implemented as an intrinsic to avoid converting to and from an
    /// integer, since the conversion inhibits certain optimizations.
    ///
    /// # Safety
    ///
    /// Unlike the `offset` intrinsic, this intrinsic does not restrict the
    /// resulting pointer to point into or one byte past the end of an allocated
    /// object, and it wraps with two's complement arithmetic. The resulting
    /// value is not necessarily valid to be used to actually access memory.
    ///
    /// The stabilized version of this intrinsic is [`pointer::wrapping_offset`].
    #[must_use = "returns a new pointer rather than modifying its argument"]
    #[rustc_const_unstable(feature = "const_ptr_offset", issue = "71499")]
    pub fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;

    /// Equivalent to the appropriate `llvm.memcpy.p0i8.0i8.*` intrinsic, with
    /// a size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    ///
    /// The volatile parameter is set to `true`, so it will not be optimized out
    /// unless size is equal to zero.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn volatile_copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: usize);
    /// Equivalent to the appropriate `llvm.memmove.p0i8.0i8.*` intrinsic, with
    /// a size of `count * size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    ///
    /// The volatile parameter is set to `true`, so it will not be optimized out
    /// unless size is equal to zero.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn volatile_copy_memory<T>(dst: *mut T, src: *const T, count: usize);
    /// Equivalent to the appropriate `llvm.memset.p0i8.*` intrinsic, with a
    /// size of `count * size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`.
    ///
    /// The volatile parameter is set to `true`, so it will not be optimized out
    /// unless size is equal to zero.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn volatile_set_memory<T>(dst: *mut T, val: u8, count: usize);

    /// Performs a volatile load from the `src` pointer.
    ///
    /// The stabilized version of this intrinsic is [`core::ptr::read_volatile`].
    pub fn volatile_load<T>(src: *const T) -> T;
    /// Performs a volatile store to the `dst` pointer.
    ///
    /// The stabilized version of this intrinsic is [`core::ptr::write_volatile`].
    pub fn volatile_store<T>(dst: *mut T, val: T);

    /// Performs a volatile load from the `src` pointer
    /// The pointer is not required to be aligned.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn unaligned_volatile_load<T>(src: *const T) -> T;
    /// Performs a volatile store to the `dst` pointer.
    /// The pointer is not required to be aligned.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn unaligned_volatile_store<T>(dst: *mut T, val: T);

    /// Returns the square root of an `f32`
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::sqrt`](../../std/primitive.f32.html#method.sqrt)
    pub fn sqrtf32(x: f32) -> f32;
    /// Returns the square root of an `f64`
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::sqrt`](../../std/primitive.f64.html#method.sqrt)
    pub fn sqrtf64(x: f64) -> f64;

    /// Raises an `f32` to an integer power.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::powi`](../../std/primitive.f32.html#method.powi)
    pub fn powif32(a: f32, x: i32) -> f32;
    /// Raises an `f64` to an integer power.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::powi`](../../std/primitive.f64.html#method.powi)
    pub fn powif64(a: f64, x: i32) -> f64;

    /// Returns the sine of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::sin`](../../std/primitive.f32.html#method.sin)
    pub fn sinf32(x: f32) -> f32;
    /// Returns the sine of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::sin`](../../std/primitive.f64.html#method.sin)
    pub fn sinf64(x: f64) -> f64;

    /// Returns the cosine of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::cos`](../../std/primitive.f32.html#method.cos)
    pub fn cosf32(x: f32) -> f32;
    /// Returns the cosine of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::cos`](../../std/primitive.f64.html#method.cos)
    pub fn cosf64(x: f64) -> f64;

    /// Raises an `f32` to an `f32` power.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::powf`](../../std/primitive.f32.html#method.powf)
    pub fn powf32(a: f32, x: f32) -> f32;
    /// Raises an `f64` to an `f64` power.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::powf`](../../std/primitive.f64.html#method.powf)
    pub fn powf64(a: f64, x: f64) -> f64;

    /// Returns the exponential of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::exp`](../../std/primitive.f32.html#method.exp)
    pub fn expf32(x: f32) -> f32;
    /// Returns the exponential of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::exp`](../../std/primitive.f64.html#method.exp)
    pub fn expf64(x: f64) -> f64;

    /// Returns 2 raised to the power of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::exp2`](../../std/primitive.f32.html#method.exp2)
    pub fn exp2f32(x: f32) -> f32;
    /// Returns 2 raised to the power of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::exp2`](../../std/primitive.f64.html#method.exp2)
    pub fn exp2f64(x: f64) -> f64;

    /// Returns the natural logarithm of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::ln`](../../std/primitive.f32.html#method.ln)
    pub fn logf32(x: f32) -> f32;
    /// Returns the natural logarithm of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::ln`](../../std/primitive.f64.html#method.ln)
    pub fn logf64(x: f64) -> f64;

    /// Returns the base 10 logarithm of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::log10`](../../std/primitive.f32.html#method.log10)
    pub fn log10f32(x: f32) -> f32;
    /// Returns the base 10 logarithm of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::log10`](../../std/primitive.f64.html#method.log10)
    pub fn log10f64(x: f64) -> f64;

    /// Returns the base 2 logarithm of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::log2`](../../std/primitive.f32.html#method.log2)
    pub fn log2f32(x: f32) -> f32;
    /// Returns the base 2 logarithm of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::log2`](../../std/primitive.f64.html#method.log2)
    pub fn log2f64(x: f64) -> f64;

    /// Returns `a * b + c` for `f32` values.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::mul_add`](../../std/primitive.f32.html#method.mul_add)
    pub fn fmaf32(a: f32, b: f32, c: f32) -> f32;
    /// Returns `a * b + c` for `f64` values.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::mul_add`](../../std/primitive.f64.html#method.mul_add)
    pub fn fmaf64(a: f64, b: f64, c: f64) -> f64;

    /// Returns the absolute value of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::abs`](../../std/primitive.f32.html#method.abs)
    pub fn fabsf32(x: f32) -> f32;
    /// Returns the absolute value of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::abs`](../../std/primitive.f64.html#method.abs)
    pub fn fabsf64(x: f64) -> f64;

    /// Returns the minimum of two `f32` values.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::min`]
    pub fn minnumf32(x: f32, y: f32) -> f32;
    /// Returns the minimum of two `f64` values.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::min`]
    pub fn minnumf64(x: f64, y: f64) -> f64;
    /// Returns the maximum of two `f32` values.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::max`]
    pub fn maxnumf32(x: f32, y: f32) -> f32;
    /// Returns the maximum of two `f64` values.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::max`]
    pub fn maxnumf64(x: f64, y: f64) -> f64;

    /// Copies the sign from `y` to `x` for `f32` values.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::copysign`](../../std/primitive.f32.html#method.copysign)
    pub fn copysignf32(x: f32, y: f32) -> f32;
    /// Copies the sign from `y` to `x` for `f64` values.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::copysign`](../../std/primitive.f64.html#method.copysign)
    pub fn copysignf64(x: f64, y: f64) -> f64;

    /// Returns the largest integer less than or equal to an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::floor`](../../std/primitive.f32.html#method.floor)
    pub fn floorf32(x: f32) -> f32;
    /// Returns the largest integer less than or equal to an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::floor`](../../std/primitive.f64.html#method.floor)
    pub fn floorf64(x: f64) -> f64;

    /// Returns the smallest integer greater than or equal to an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::ceil`](../../std/primitive.f32.html#method.ceil)
    pub fn ceilf32(x: f32) -> f32;
    /// Returns the smallest integer greater than or equal to an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::ceil`](../../std/primitive.f64.html#method.ceil)
    pub fn ceilf64(x: f64) -> f64;

    /// Returns the integer part of an `f32`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::trunc`](../../std/primitive.f32.html#method.trunc)
    pub fn truncf32(x: f32) -> f32;
    /// Returns the integer part of an `f64`.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::trunc`](../../std/primitive.f64.html#method.trunc)
    pub fn truncf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`. May raise an inexact floating-point exception
    /// if the argument is not an integer.
    pub fn rintf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`. May raise an inexact floating-point exception
    /// if the argument is not an integer.
    pub fn rintf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn nearbyintf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn nearbyintf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`. Rounds half-way cases away from zero.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f32::round`](../../std/primitive.f32.html#method.round)
    pub fn roundf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`. Rounds half-way cases away from zero.
    ///
    /// The stabilized version of this intrinsic is
    /// [`f64::round`](../../std/primitive.f64.html#method.round)
    pub fn roundf64(x: f64) -> f64;

    /// Float addition that allows optimizations based on algebraic rules.
    /// May assume inputs are finite.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn fadd_fast<T: Copy>(a: T, b: T) -> T;

    /// Float subtraction that allows optimizations based on algebraic rules.
    /// May assume inputs are finite.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn fsub_fast<T: Copy>(a: T, b: T) -> T;

    /// Float multiplication that allows optimizations based on algebraic rules.
    /// May assume inputs are finite.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn fmul_fast<T: Copy>(a: T, b: T) -> T;

    /// Float division that allows optimizations based on algebraic rules.
    /// May assume inputs are finite.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn fdiv_fast<T: Copy>(a: T, b: T) -> T;

    /// Float remainder that allows optimizations based on algebraic rules.
    /// May assume inputs are finite.
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn frem_fast<T: Copy>(a: T, b: T) -> T;

    /// Convert with LLVM’s fptoui/fptosi, which may return undef for values out of range
    /// (<https://github.com/rust-lang/rust/issues/10184>)
    ///
    /// Stabilized as [`f32::to_int_unchecked`] and [`f64::to_int_unchecked`].
    pub fn float_to_int_unchecked<Float: Copy, Int: Copy>(value: Float) -> Int;

    /// Returns the number of bits set in an integer type `T`
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `count_ones` method. For example,
    /// [`u32::count_ones`]
    #[rustc_const_stable(feature = "const_ctpop", since = "1.40.0")]
    pub fn ctpop<T: Copy>(x: T) -> T;

    /// Returns the number of leading unset bits (zeroes) in an integer type `T`.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `leading_zeros` method. For example,
    /// [`u32::leading_zeros`]
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::ctlz;
    ///
    /// let x = 0b0001_1100_u8;
    /// let num_leading = ctlz(x);
    /// assert_eq!(num_leading, 3);
    /// ```
    ///
    /// An `x` with value `0` will return the bit width of `T`.
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::ctlz;
    ///
    /// let x = 0u16;
    /// let num_leading = ctlz(x);
    /// assert_eq!(num_leading, 16);
    /// ```
    #[rustc_const_stable(feature = "const_ctlz", since = "1.40.0")]
    pub fn ctlz<T: Copy>(x: T) -> T;

    /// Like `ctlz`, but extra-unsafe as it returns `undef` when
    /// given an `x` with value `0`.
    ///
    /// This intrinsic does not have a stable counterpart.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::ctlz_nonzero;
    ///
    /// let x = 0b0001_1100_u8;
    /// let num_leading = unsafe { ctlz_nonzero(x) };
    /// assert_eq!(num_leading, 3);
    /// ```
    #[rustc_const_stable(feature = "constctlz", since = "1.50.0")]
    pub fn ctlz_nonzero<T: Copy>(x: T) -> T;

    /// Returns the number of trailing unset bits (zeroes) in an integer type `T`.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `trailing_zeros` method. For example,
    /// [`u32::trailing_zeros`]
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::cttz;
    ///
    /// let x = 0b0011_1000_u8;
    /// let num_trailing = cttz(x);
    /// assert_eq!(num_trailing, 3);
    /// ```
    ///
    /// An `x` with value `0` will return the bit width of `T`:
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::cttz;
    ///
    /// let x = 0u16;
    /// let num_trailing = cttz(x);
    /// assert_eq!(num_trailing, 16);
    /// ```
    #[rustc_const_stable(feature = "const_cttz", since = "1.40.0")]
    pub fn cttz<T: Copy>(x: T) -> T;

    /// Like `cttz`, but extra-unsafe as it returns `undef` when
    /// given an `x` with value `0`.
    ///
    /// This intrinsic does not have a stable counterpart.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(core_intrinsics)]
    ///
    /// use std::intrinsics::cttz_nonzero;
    ///
    /// let x = 0b0011_1000_u8;
    /// let num_trailing = unsafe { cttz_nonzero(x) };
    /// assert_eq!(num_trailing, 3);
    /// ```
    #[rustc_const_stable(feature = "const_cttz", since = "1.53.0")]
    pub fn cttz_nonzero<T: Copy>(x: T) -> T;

    /// Reverses the bytes in an integer type `T`.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `swap_bytes` method. For example,
    /// [`u32::swap_bytes`]
    #[rustc_const_stable(feature = "const_bswap", since = "1.40.0")]
    pub fn bswap<T: Copy>(x: T) -> T;

    /// Reverses the bits in an integer type `T`.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `reverse_bits` method. For example,
    /// [`u32::reverse_bits`]
    #[rustc_const_stable(feature = "const_bitreverse", since = "1.40.0")]
    pub fn bitreverse<T: Copy>(x: T) -> T;

    /// Performs checked integer addition.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `overflowing_add` method. For example,
    /// [`u32::overflowing_add`]
    #[rustc_const_stable(feature = "const_int_overflow", since = "1.40.0")]
    pub fn add_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

    /// Performs checked integer subtraction
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `overflowing_sub` method. For example,
    /// [`u32::overflowing_sub`]
    #[rustc_const_stable(feature = "const_int_overflow", since = "1.40.0")]
    pub fn sub_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

    /// Performs checked integer multiplication
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `overflowing_mul` method. For example,
    /// [`u32::overflowing_mul`]
    #[rustc_const_stable(feature = "const_int_overflow", since = "1.40.0")]
    pub fn mul_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

    /// Performs an exact division, resulting in undefined behavior where
    /// `x % y != 0` or `y == 0` or `x == T::MIN && y == -1`
    ///
    /// This intrinsic does not have a stable counterpart.
    pub fn exact_div<T: Copy>(x: T, y: T) -> T;

    /// Performs an unchecked division, resulting in undefined behavior
    /// where `y == 0` or `x == T::MIN && y == -1`
    ///
    /// Safe wrappers for this intrinsic are available on the integer
    /// primitives via the `checked_div` method. For example,
    /// [`u32::checked_div`]
    #[rustc_const_stable(feature = "const_int_unchecked_arith", since = "1.52.0")]
    pub fn unchecked_div<T: Copy>(x: T, y: T) -> T;
    /// Returns the remainder of an unchecked division, resulting in
    /// undefined behavior when `y == 0` or `x == T::MIN && y == -1`
    ///
    /// Safe wrappers for this intrinsic are available on the integer
    /// primitives via the `checked_rem` method. For example,
    /// [`u32::checked_rem`]
    #[rustc_const_stable(feature = "const_int_unchecked_arith", since = "1.52.0")]
    pub fn unchecked_rem<T: Copy>(x: T, y: T) -> T;

    /// Performs an unchecked left shift, resulting in undefined behavior when
    /// `y < 0` or `y >= N`, where N is the width of T in bits.
    ///
    /// Safe wrappers for this intrinsic are available on the integer
    /// primitives via the `checked_shl` method. For example,
    /// [`u32::checked_shl`]
    #[rustc_const_stable(feature = "const_int_unchecked", since = "1.40.0")]
    pub fn unchecked_shl<T: Copy>(x: T, y: T) -> T;
    /// Performs an unchecked right shift, resulting in undefined behavior when
    /// `y < 0` or `y >= N`, where N is the width of T in bits.
    ///
    /// Safe wrappers for this intrinsic are available on the integer
    /// primitives via the `checked_shr` method. For example,
    /// [`u32::checked_shr`]
    #[rustc_const_stable(feature = "const_int_unchecked", since = "1.40.0")]
    pub fn unchecked_shr<T: Copy>(x: T, y: T) -> T;

    /// Returns the result of an unchecked addition, resulting in
    /// undefined behavior when `x + y > T::MAX` or `x + y < T::MIN`.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_int_unchecked_arith", issue = "none")]
    pub fn unchecked_add<T: Copy>(x: T, y: T) -> T;

    /// Returns the result of an unchecked subtraction, resulting in
    /// undefined behavior when `x - y > T::MAX` or `x - y < T::MIN`.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_int_unchecked_arith", issue = "none")]
    pub fn unchecked_sub<T: Copy>(x: T, y: T) -> T;

    /// Returns the result of an unchecked multiplication, resulting in
    /// undefined behavior when `x * y > T::MAX` or `x * y < T::MIN`.
    ///
    /// This intrinsic does not have a stable counterpart.
    #[rustc_const_unstable(feature = "const_int_unchecked_arith", issue = "none")]
    pub fn unchecked_mul<T: Copy>(x: T, y: T) -> T;

    /// Performs rotate left.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `rotate_left` method. For example,
    /// [`u32::rotate_left`]
    #[rustc_const_stable(feature = "const_int_rotate", since = "1.40.0")]
    pub fn rotate_left<T: Copy>(x: T, y: T) -> T;

    /// Performs rotate right.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `rotate_right` method. For example,
    /// [`u32::rotate_right`]
    #[rustc_const_stable(feature = "const_int_rotate", since = "1.40.0")]
    pub fn rotate_right<T: Copy>(x: T, y: T) -> T;

    /// Returns (a + b) mod 2<sup>N</sup>, where N is the width of T in bits.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `wrapping_add` method. For example,
    /// [`u32::wrapping_add`]
    #[rustc_const_stable(feature = "const_int_wrapping", since = "1.40.0")]
    pub fn wrapping_add<T: Copy>(a: T, b: T) -> T;
    /// Returns (a - b) mod 2<sup>N</sup>, where N is the width of T in bits.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `wrapping_sub` method. For example,
    /// [`u32::wrapping_sub`]
    #[rustc_const_stable(feature = "const_int_wrapping", since = "1.40.0")]
    pub fn wrapping_sub<T: Copy>(a: T, b: T) -> T;
    /// Returns (a * b) mod 2<sup>N</sup>, where N is the width of T in bits.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `wrapping_mul` method. For example,
    /// [`u32::wrapping_mul`]
    #[rustc_const_stable(feature = "const_int_wrapping", since = "1.40.0")]
    pub fn wrapping_mul<T: Copy>(a: T, b: T) -> T;

    /// Computes `a + b`, saturating at numeric bounds.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `saturating_add` method. For example,
    /// [`u32::saturating_add`]
    #[rustc_const_stable(feature = "const_int_saturating", since = "1.40.0")]
    pub fn saturating_add<T: Copy>(a: T, b: T) -> T;
    /// Computes `a - b`, saturating at numeric bounds.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized versions of this intrinsic are available on the integer
    /// primitives via the `saturating_sub` method. For example,
    /// [`u32::saturating_sub`]
    #[rustc_const_stable(feature = "const_int_saturating", since = "1.40.0")]
    pub fn saturating_sub<T: Copy>(a: T, b: T) -> T;

    /// Returns the value of the discriminant for the variant in 'v';
    /// if `T` has no discriminant, returns `0`.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The stabilized version of this intrinsic is [`core::mem::discriminant`].
    #[rustc_const_unstable(feature = "const_discriminant", issue = "69821")]
    pub fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;

    /// Returns the number of variants of the type `T` cast to a `usize`;
    /// if `T` has no variants, returns `0`. Uninhabited variants will be counted.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    ///
    /// The to-be-stabilized version of this intrinsic is [`mem::variant_count`].
    #[rustc_const_unstable(feature = "variant_count", issue = "73662")]
    pub fn variant_count<T>() -> usize;

    /// Rust's "try catch" construct which invokes the function pointer `try_fn`
    /// with the data pointer `data`.
    ///
    /// The third argument is a function called if a panic occurs. This function
    /// takes the data pointer and a pointer to the target-specific exception
    /// object that was caught. For more information see the compiler's
    /// source as well as std's catch implementation.
    pub fn r#try(try_fn: fn(*mut u8), data: *mut u8, catch_fn: fn(*mut u8, *mut u8)) -> i32;

    /// Emits a `!nontemporal` store according to LLVM (see their docs).
    /// Probably will never become stable.
    pub fn nontemporal_store<T>(ptr: *mut T, val: T);

    /// See documentation of `<*const T>::offset_from` for details.
    #[rustc_const_unstable(feature = "const_ptr_offset_from", issue = "41079")]
    pub fn ptr_offset_from<T>(ptr: *const T, base: *const T) -> isize;

    /// See documentation of `<*const T>::guaranteed_eq` for details.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    pub fn ptr_guaranteed_eq<T>(ptr: *const T, other: *const T) -> bool;

    /// See documentation of `<*const T>::guaranteed_ne` for details.
    ///
    /// Note that, unlike most intrinsics, this is safe to call;
    /// it does not require an `unsafe` block.
    /// Therefore, implementations must not require the user to uphold
    /// any safety invariants.
    #[rustc_const_unstable(feature = "const_raw_ptr_comparison", issue = "53020")]
    pub fn ptr_guaranteed_ne<T>(ptr: *const T, other: *const T) -> bool;

    /// Allocate at compile time. Should not be called at runtime.
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    pub fn const_allocate(size: usize, align: usize) -> *mut u8;
}

// Some functions are defined here because they accidentally got made
// available in this module on stable. See <https://github.com/rust-lang/rust/issues/15702>.
// (`transmute` also falls into this category, but it cannot be wrapped due to the
// check that `T` and `U` have the same size.)

/// Checks whether `ptr` is properly aligned with respect to
/// `align_of::<T>()`.
pub(crate) fn is_aligned_and_not_null<T>(ptr: *const T) -> bool {
    !ptr.is_null() && ptr as usize % mem::align_of::<T>() == 0
}

/// Copies `count * size_of::<T>()` bytes from `src` to `dst`. The source
/// and destination must *not* overlap.
///
/// For regions of memory which might overlap, use [`copy`] instead.
///
/// `copy_nonoverlapping` is semantically equivalent to C's [`memcpy`], but
/// with the argument order swapped.
///
/// [`memcpy`]: https://en.cppreference.com/w/c/string/byte/memcpy
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count * size_of::<T>()` bytes.
///
/// * `dst` must be [valid] for writes of `count * size_of::<T>()` bytes.
///
/// * Both `src` and `dst` must be properly aligned.
///
/// * The region of memory beginning at `src` with a size of `count *
///   size_of::<T>()` bytes must *not* overlap with the region of memory
///   beginning at `dst` with the same size.
///
/// Like [`read`], `copy_nonoverlapping` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using *both* the values
/// in the region beginning at `*src` and the region beginning at `*dst` can
/// [violate memory safety][read-ownership].
///
/// Note that even if the effectively copied size (`count * size_of::<T>()`) is
/// `0`, the pointers must be non-null and properly aligned.
///
/// [`read`]: crate::ptr::read
/// [read-ownership]: crate::ptr::read#ownership-of-the-returned-value
/// [valid]: crate::ptr#safety
///
/// # Examples
///
/// Manually implement [`Vec::append`]:
///
/// ```
/// use std::ptr;
///
/// /// Moves all the elements of `src` into `dst`, leaving `src` empty.
/// fn append<T>(dst: &mut Vec<T>, src: &mut Vec<T>) {
///     let src_len = src.len();
///     let dst_len = dst.len();
///
///     // Ensure that `dst` has enough capacity to hold all of `src`.
///     dst.reserve(src_len);
///
///     unsafe {
///         // The call to offset is always safe because `Vec` will never
///         // allocate more than `isize::MAX` bytes.
///         let dst_ptr = dst.as_mut_ptr().offset(dst_len as isize);
///         let src_ptr = src.as_ptr();
///
///         // Truncate `src` without dropping its contents. We do this first,
///         // to avoid problems in case something further down panics.
///         src.set_len(0);
///
///         // The two regions cannot overlap because mutable references do
///         // not alias, and two different vectors cannot own the same
///         // memory.
///         ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_len);
///
///         // Notify `dst` that it now holds the contents of `src`.
///         dst.set_len(dst_len + src_len);
///     }
/// }
///
/// let mut a = vec!['r'];
/// let mut b = vec!['u', 's', 't'];
///
/// append(&mut a, &mut b);
///
/// assert_eq!(a, &['r', 'u', 's', 't']);
/// assert!(b.is_empty());
/// ```
///
/// [`Vec::append`]: ../../std/vec/struct.Vec.html#method.append
#[doc(alias = "memcpy")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
#[inline]
pub const unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize) {
    extern "rust-intrinsic" {
        #[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
        pub fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
    }

    // FIXME: Perform these checks only at run time
    /*if cfg!(debug_assertions)
        && !(is_aligned_and_not_null(src)
            && is_aligned_and_not_null(dst)
            && is_nonoverlapping(src, dst, count))
    {
        // Not panicking to keep codegen impact smaller.
        abort();
    }*/

    // SAFETY: the safety contract for `copy_nonoverlapping` must be
    // upheld by the caller.
    unsafe { copy_nonoverlapping(src, dst, count) }
}

/// Copies `count * size_of::<T>()` bytes from `src` to `dst`. The source
/// and destination may overlap.
///
/// If the source and destination will *never* overlap,
/// [`copy_nonoverlapping`] can be used instead.
///
/// `copy` is semantically equivalent to C's [`memmove`], but with the argument
/// order swapped. Copying takes place as if the bytes were copied from `src`
/// to a temporary array and then copied from the array to `dst`.
///
/// [`memmove`]: https://en.cppreference.com/w/c/string/byte/memmove
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count * size_of::<T>()` bytes.
///
/// * `dst` must be [valid] for writes of `count * size_of::<T>()` bytes.
///
/// * Both `src` and `dst` must be properly aligned.
///
/// Like [`read`], `copy` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the values
/// in the region beginning at `*src` and the region beginning at `*dst` can
/// [violate memory safety][read-ownership].
///
/// Note that even if the effectively copied size (`count * size_of::<T>()`) is
/// `0`, the pointers must be non-null and properly aligned.
///
/// [`read`]: crate::ptr::read
/// [read-ownership]: crate::ptr::read#ownership-of-the-returned-value
/// [valid]: crate::ptr#safety
///
/// # Examples
///
/// Efficiently create a Rust vector from an unsafe buffer:
///
/// ```
/// use std::ptr;
///
/// /// # Safety
/// ///
/// /// * `ptr` must be correctly aligned for its type and non-zero.
/// /// * `ptr` must be valid for reads of `elts` contiguous elements of type `T`.
/// /// * Those elements must not be used after calling this function unless `T: Copy`.
/// # #[allow(dead_code)]
/// unsafe fn from_buf_raw<T>(ptr: *const T, elts: usize) -> Vec<T> {
///     let mut dst = Vec::with_capacity(elts);
///
///     // SAFETY: Our precondition ensures the source is aligned and valid,
///     // and `Vec::with_capacity` ensures that we have usable space to write them.
///     ptr::copy(ptr, dst.as_mut_ptr(), elts);
///
///     // SAFETY: We created it with this much capacity earlier,
///     // and the previous `copy` has initialized these elements.
///     dst.set_len(elts);
///     dst
/// }
/// ```
#[doc(alias = "memmove")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
#[inline]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    extern "rust-intrinsic" {
        #[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
        fn copy<T>(src: *const T, dst: *mut T, count: usize);
    }

    // FIXME: Perform these checks only at run time
    /*if cfg!(debug_assertions) && !(is_aligned_and_not_null(src) && is_aligned_and_not_null(dst)) {
        // Not panicking to keep codegen impact smaller.
        abort();
    }*/

    // SAFETY: the safety contract for `copy` must be upheld by the caller.
    unsafe { copy(src, dst, count) }
}

/// Sets `count * size_of::<T>()` bytes of memory starting at `dst` to
/// `val`.
///
/// `write_bytes` is similar to C's [`memset`], but sets `count *
/// size_of::<T>()` bytes to `val`.
///
/// [`memset`]: https://en.cppreference.com/w/c/string/byte/memset
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes of `count * size_of::<T>()` bytes.
///
/// * `dst` must be properly aligned.
///
/// Additionally, the caller must ensure that writing `count *
/// size_of::<T>()` bytes to the given region of memory results in a valid
/// value of `T`. Using a region of memory typed as a `T` that contains an
/// invalid value of `T` is undefined behavior.
///
/// Note that even if the effectively copied size (`count * size_of::<T>()`) is
/// `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: crate::ptr#safety
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::ptr;
///
/// let mut vec = vec![0u32; 4];
/// unsafe {
///     let vec_ptr = vec.as_mut_ptr();
///     ptr::write_bytes(vec_ptr, 0xfe, 2);
/// }
/// assert_eq!(vec, [0xfefefefe, 0xfefefefe, 0, 0]);
/// ```
///
/// Creating an invalid value:
///
/// ```
/// use std::ptr;
///
/// let mut v = Box::new(0i32);
///
/// unsafe {
///     // Leaks the previously held value by overwriting the `Box<T>` with
///     // a null pointer.
///     ptr::write_bytes(&mut v as *mut Box<i32>, 0, 1);
/// }
///
/// // At this point, using or dropping `v` results in undefined behavior.
/// // drop(v); // ERROR
///
/// // Even leaking `v` "uses" it, and hence is undefined behavior.
/// // mem::forget(v); // ERROR
///
/// // In fact, `v` is invalid according to basic type layout invariants, so *any*
/// // operation touching it is undefined behavior.
/// // let v2 = v; // ERROR
///
/// unsafe {
///     // Let us instead put in a valid value
///     ptr::write(&mut v as *mut Box<i32>, Box::new(42i32));
/// }
///
/// // Now the box is fine
/// assert_eq!(*v, 42);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[inline]
pub unsafe fn write_bytes<T>(dst: *mut T, val: u8, count: usize) {
    extern "rust-intrinsic" {
        fn write_bytes<T>(dst: *mut T, val: u8, count: usize);
    }

    debug_assert!(is_aligned_and_not_null(dst), "attempt to write to unaligned or null pointer");

    // SAFETY: the safety contract for `write_bytes` must be upheld by the caller.
    unsafe { write_bytes(dst, val, count) }
}
