//! Compiler intrinsics.
//!
//! These are the imports making intrinsics available to Rust code. The actual implementations live in the compiler.
//! Some of these intrinsics are lowered to MIR in <https://github.com/rust-lang/rust/blob/master/compiler/rustc_mir_transform/src/lower_intrinsics.rs>.
//! The remaining intrinsics are implemented for the LLVM backend in <https://github.com/rust-lang/rust/blob/master/compiler/rustc_codegen_ssa/src/mir/intrinsic.rs>
//! and <https://github.com/rust-lang/rust/blob/master/compiler/rustc_codegen_llvm/src/intrinsic.rs>,
//! and for const evaluation in <https://github.com/rust-lang/rust/blob/master/compiler/rustc_const_eval/src/interpret/intrinsics.rs>.
//!
//! # Const intrinsics
//!
//! In order to make an intrinsic unstable usable at compile-time, copy the implementation from
//! <https://github.com/rust-lang/miri/blob/master/src/intrinsics> to
//! <https://github.com/rust-lang/rust/blob/master/compiler/rustc_const_eval/src/interpret/intrinsics.rs>
//! and make the intrinsic declaration below a `const fn`. This should be done in coordination with
//! wg-const-eval.
//!
//! If an intrinsic is supposed to be used from a `const fn` with a `rustc_const_stable` attribute,
//! `#[rustc_intrinsic_const_stable_indirect]` needs to be added to the intrinsic. Such a change requires
//! T-lang approval, because it may bake a feature into the language that cannot be replicated in
//! user code without compiler support.
//!
//! # Volatiles
//!
//! The volatile intrinsics provide operations intended to act on I/O
//! memory, which are guaranteed to not be reordered by the compiler
//! across other volatile intrinsics. See [`read_volatile`][ptr::read_volatile]
//! and [`write_volatile`][ptr::write_volatile].
//!
//! # Atomics
//!
//! The atomic intrinsics provide common atomic operations on machine
//! words, with multiple possible memory orderings. See the
//! [atomic types][crate::sync::atomic] docs for details.
//!
//! # Unwinding
//!
//! Rust intrinsics may, in general, unwind. If an intrinsic can never unwind, add the
//! `#[rustc_nounwind]` attribute so that the compiler can make use of this fact.
//!
//! However, even for intrinsics that may unwind, rustc assumes that a Rust intrinsics will never
//! initiate a foreign (non-Rust) unwind, and thus for panic=abort we can always assume that these
//! intrinsics cannot unwind.

#![unstable(
    feature = "core_intrinsics",
    reason = "intrinsics are unlikely to ever be stabilized, instead \
                      they should be used through stabilized interfaces \
                      in the rest of the standard library",
    issue = "none"
)]
#![allow(missing_docs)]

use crate::marker::{DiscriminantKind, Tuple};
use crate::ptr;

pub mod fallback;
pub mod mir;
pub mod simd;

// These imports are used for simplifying intra-doc links
#[allow(unused_imports)]
#[cfg(all(target_has_atomic = "8", target_has_atomic = "32", target_has_atomic = "ptr"))]
use crate::sync::atomic::{self, AtomicBool, AtomicI32, AtomicIsize, AtomicU32, Ordering};

// N.B., these intrinsics take raw pointers because they mutate aliased
// memory, which is not valid for either `&` or `&mut`.

/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Relaxed`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_relaxed_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Relaxed`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_relaxed_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Relaxed`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_relaxed_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Acquire`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acquire_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Acquire`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acquire_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Acquire`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acquire_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Release`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_release_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Release`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_release_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::Release`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_release_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acqrel_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acqrel_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_acqrel_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::SeqCst`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_seqcst_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::SeqCst`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_seqcst_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange` method by passing
/// [`Ordering::SeqCst`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchg_seqcst_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);

/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Relaxed`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_relaxed_relaxed<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Relaxed`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_relaxed_acquire<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Relaxed`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_relaxed_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Acquire`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acquire_relaxed<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Acquire`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acquire_acquire<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Acquire`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acquire_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Release`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_release_relaxed<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Release`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_release_acquire<T: Copy>(
    _dst: *mut T,
    _old: T,
    _src: T,
) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::Release`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_release_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acqrel_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acqrel_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::AcqRel`] and [`Ordering::SeqCst`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_acqrel_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::SeqCst`] and [`Ordering::Relaxed`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_seqcst_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::SeqCst`] and [`Ordering::Acquire`] as the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_seqcst_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
/// Stores a value if the current value is the same as the `old` value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `compare_exchange_weak` method by passing
/// [`Ordering::SeqCst`] as both the success and failure parameters.
/// For example, [`AtomicBool::compare_exchange_weak`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_cxchgweak_seqcst_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);

/// Loads the current value of the pointer.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `load` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::load`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_load_seqcst<T: Copy>(src: *const T) -> T;
/// Loads the current value of the pointer.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `load` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::load`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_load_acquire<T: Copy>(src: *const T) -> T;
/// Loads the current value of the pointer.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `load` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::load`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_load_relaxed<T: Copy>(src: *const T) -> T;

/// Stores the value at the specified memory location.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `store` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::store`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_store_seqcst<T: Copy>(dst: *mut T, val: T);
/// Stores the value at the specified memory location.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `store` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::store`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_store_release<T: Copy>(dst: *mut T, val: T);
/// Stores the value at the specified memory location.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `store` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::store`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_store_relaxed<T: Copy>(dst: *mut T, val: T);

/// Stores the value at the specified memory location, returning the old value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `swap` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::swap`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xchg_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Stores the value at the specified memory location, returning the old value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `swap` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::swap`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xchg_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Stores the value at the specified memory location, returning the old value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `swap` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::swap`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xchg_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Stores the value at the specified memory location, returning the old value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `swap` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::swap`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xchg_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Stores the value at the specified memory location, returning the old value.
/// `T` must be an integer or pointer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `swap` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::swap`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xchg_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Adds to the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_add` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicIsize::fetch_add`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xadd_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Adds to the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_add` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicIsize::fetch_add`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xadd_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Adds to the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_add` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicIsize::fetch_add`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xadd_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Adds to the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_add` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicIsize::fetch_add`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xadd_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Adds to the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_add` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicIsize::fetch_add`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xadd_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Subtract from the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_sub` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xsub_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Subtract from the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_sub` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xsub_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Subtract from the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_sub` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xsub_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Subtract from the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_sub` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xsub_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Subtract from the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_sub` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicIsize::fetch_sub`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xsub_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Bitwise and with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_and` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_and`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_and_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise and with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_and` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_and`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_and_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise and with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_and` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_and`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_and_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise and with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_and` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_and`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_and_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise and with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_and` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_and`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_and_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Bitwise nand with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`AtomicBool`] type via the `fetch_nand` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_nand`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_nand_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise nand with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`AtomicBool`] type via the `fetch_nand` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_nand`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_nand_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise nand with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`AtomicBool`] type via the `fetch_nand` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_nand`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_nand_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise nand with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`AtomicBool`] type via the `fetch_nand` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_nand`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_nand_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise nand with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`AtomicBool`] type via the `fetch_nand` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_nand`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_nand_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Bitwise or with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_or` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_or`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_or_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise or with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_or` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_or`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_or_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise or with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_or` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_or`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_or_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise or with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_or` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_or`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_or_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise or with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_or` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_or`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_or_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Bitwise xor with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_xor` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicBool::fetch_xor`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xor_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise xor with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_xor` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicBool::fetch_xor`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xor_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise xor with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_xor` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicBool::fetch_xor`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xor_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise xor with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_xor` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicBool::fetch_xor`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xor_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Bitwise xor with the current value, returning the previous value.
/// `T` must be an integer or pointer type.
/// If `T` is a pointer type, the provenance of `src` is ignored: both the return value and the new
/// value stored at `*dst` will have the provenance of the old value stored there.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] types via the `fetch_xor` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicBool::fetch_xor`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_xor_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Maximum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_max` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicI32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_max_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_max` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicI32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_max_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_max` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicI32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_max_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_max` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicI32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_max_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_max` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicI32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_max_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Minimum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_min` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicI32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_min_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_min` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicI32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_min_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_min` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicI32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_min_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_min` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicI32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_min_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using a signed comparison.
/// `T` must be a signed integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] signed integer types via the `fetch_min` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicI32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_min_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Minimum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_min` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicU32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umin_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_min` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicU32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umin_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_min` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicU32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umin_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_min` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicU32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umin_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Minimum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_min` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicU32::fetch_min`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umin_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// Maximum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_max` method by passing
/// [`Ordering::SeqCst`] as the `order`. For example, [`AtomicU32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umax_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_max` method by passing
/// [`Ordering::Acquire`] as the `order`. For example, [`AtomicU32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umax_acquire<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_max` method by passing
/// [`Ordering::Release`] as the `order`. For example, [`AtomicU32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umax_release<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_max` method by passing
/// [`Ordering::AcqRel`] as the `order`. For example, [`AtomicU32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umax_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
/// Maximum with the current value using an unsigned comparison.
/// `T` must be an unsigned integer type.
///
/// The stabilized version of this intrinsic is available on the
/// [`atomic`] unsigned integer types via the `fetch_max` method by passing
/// [`Ordering::Relaxed`] as the `order`. For example, [`AtomicU32::fetch_max`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_umax_relaxed<T: Copy>(dst: *mut T, src: T) -> T;

/// An atomic fence.
///
/// The stabilized version of this intrinsic is available in
/// [`atomic::fence`] by passing [`Ordering::SeqCst`]
/// as the `order`.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_fence_seqcst();
/// An atomic fence.
///
/// The stabilized version of this intrinsic is available in
/// [`atomic::fence`] by passing [`Ordering::Acquire`]
/// as the `order`.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_fence_acquire();
/// An atomic fence.
///
/// The stabilized version of this intrinsic is available in
/// [`atomic::fence`] by passing [`Ordering::Release`]
/// as the `order`.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_fence_release();
/// An atomic fence.
///
/// The stabilized version of this intrinsic is available in
/// [`atomic::fence`] by passing [`Ordering::AcqRel`]
/// as the `order`.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_fence_acqrel();

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
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_singlethreadfence_seqcst();
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
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_singlethreadfence_acquire();
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
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_singlethreadfence_release();
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
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn atomic_singlethreadfence_acqrel();

/// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
/// if supported; otherwise, it is a no-op.
/// Prefetches have no effect on the behavior of the program but can change its performance
/// characteristics.
///
/// The `locality` argument must be a constant integer and is a temporal locality specifier
/// ranging from (0) - no locality, to (3) - extremely local keep in cache.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn prefetch_read_data<T>(data: *const T, locality: i32);
/// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
/// if supported; otherwise, it is a no-op.
/// Prefetches have no effect on the behavior of the program but can change its performance
/// characteristics.
///
/// The `locality` argument must be a constant integer and is a temporal locality specifier
/// ranging from (0) - no locality, to (3) - extremely local keep in cache.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn prefetch_write_data<T>(data: *const T, locality: i32);
/// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
/// if supported; otherwise, it is a no-op.
/// Prefetches have no effect on the behavior of the program but can change its performance
/// characteristics.
///
/// The `locality` argument must be a constant integer and is a temporal locality specifier
/// ranging from (0) - no locality, to (3) - extremely local keep in cache.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn prefetch_read_instruction<T>(data: *const T, locality: i32);
/// The `prefetch` intrinsic is a hint to the code generator to insert a prefetch instruction
/// if supported; otherwise, it is a no-op.
/// Prefetches have no effect on the behavior of the program but can change its performance
/// characteristics.
///
/// The `locality` argument must be a constant integer and is a temporal locality specifier
/// ranging from (0) - no locality, to (3) - extremely local keep in cache.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn prefetch_write_instruction<T>(data: *const T, locality: i32);

/// Executes a breakpoint trap, for inspection by a debugger.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub fn breakpoint();

/// Magic intrinsic that derives its meaning from attributes
/// attached to the function.
///
/// For example, dataflow uses this to inject static assertions so
/// that `rustc_peek(potentially_uninitialized)` would actually
/// double-check that dataflow did indeed compute that it is
/// uninitialized at that point in the control flow.
///
/// This intrinsic should not be used outside of the compiler.
#[rustc_nounwind]
#[rustc_intrinsic]
pub fn rustc_peek<T>(_: T) -> T;

/// Aborts the execution of the process.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// [`std::process::abort`](../../std/process/fn.abort.html) is to be preferred if possible,
/// as its behavior is more user-friendly and more stable.
///
/// The current implementation of `intrinsics::abort` is to invoke an invalid instruction,
/// on most platforms.
/// On Unix, the
/// process will probably terminate with a signal like `SIGABRT`, `SIGILL`, `SIGTRAP`, `SIGSEGV` or
/// `SIGBUS`.  The precise behavior is not guaranteed and not stable.
#[rustc_nounwind]
#[rustc_intrinsic]
pub fn abort() -> !;

/// Informs the optimizer that this point in the code is not reachable,
/// enabling further optimizations.
///
/// N.B., this is very different from the `unreachable!()` macro: Unlike the
/// macro, which panics when it is executed, it is *undefined behavior* to
/// reach code marked with this function.
///
/// The stabilized version of this intrinsic is [`core::hint::unreachable_unchecked`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unreachable() -> !;

/// Informs the optimizer that a condition is always true.
/// If the condition is false, the behavior is undefined.
///
/// No code is generated for this intrinsic, but the optimizer will try
/// to preserve it (and its condition) between passes, which may interfere
/// with optimization of surrounding code and reduce performance. It should
/// not be used if the invariant can be discovered by the optimizer on its
/// own, or if it does not enable any significant optimizations.
///
/// The stabilized version of this intrinsic is [`core::hint::assert_unchecked`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const unsafe fn assume(b: bool) {
    if !b {
        // SAFETY: the caller must guarantee the argument is never `false`
        unsafe { unreachable() }
    }
}

/// Hints to the compiler that current code path is cold.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// This intrinsic does not have a stable counterpart.
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
#[rustc_nounwind]
#[miri::intrinsic_fallback_is_spec]
#[cold]
pub const fn cold_path() {}

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
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_nounwind]
#[inline(always)]
pub const fn likely(b: bool) -> bool {
    if b {
        true
    } else {
        cold_path();
        false
    }
}

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
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_nounwind]
#[inline(always)]
pub const fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
        true
    } else {
        false
    }
}

/// Returns either `true_val` or `false_val` depending on condition `b` with a
/// hint to the compiler that this condition is unlikely to be correctly
/// predicted by a CPU's branch predictor (e.g. a binary search).
///
/// This is otherwise functionally equivalent to `if b { true_val } else { false_val }`.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The public form of this instrinsic is [`core::hint::select_unpredictable`].
/// However unlike the public form, the intrinsic will not drop the value that
/// is not selected.
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
#[rustc_nounwind]
#[miri::intrinsic_fallback_is_spec]
#[inline]
pub fn select_unpredictable<T>(b: bool, true_val: T, false_val: T) -> T {
    if b { true_val } else { false_val }
}

/// A guard for unsafe functions that cannot ever be executed if `T` is uninhabited:
/// This will statically either panic, or do nothing.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn assert_inhabited<T>();

/// A guard for unsafe functions that cannot ever be executed if `T` does not permit
/// zero-initialization: This will statically either panic, or do nothing.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn assert_zero_valid<T>();

/// A guard for `std::mem::uninitialized`. This will statically either panic, or do nothing.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn assert_mem_uninitialized_valid<T>();

/// Gets a reference to a static `Location` indicating where it was called.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// Consider using [`core::panic::Location::caller`] instead.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn caller_location() -> &'static crate::panic::Location<'static>;

/// Moves a value out of scope without running drop glue.
///
/// This exists solely for [`crate::mem::forget_unsized`]; normal `forget` uses
/// `ManuallyDrop` instead.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn forget<T: ?Sized>(_: T);

/// Reinterprets the bits of a value of one type as another type.
///
/// Both types must have the same size. Compilation will fail if this is not guaranteed.
///
/// `transmute` is semantically equivalent to a bitwise move of one type
/// into another. It copies the bits from the source value into the
/// destination value, then forgets the original. Note that source and destination
/// are passed by-value, which means if `Src` or `Dst` contain padding, that padding
/// is *not* guaranteed to be preserved by `transmute`.
///
/// Both the argument and the result must be [valid](../../nomicon/what-unsafe-does.html) at
/// their given type. Violating this condition leads to [undefined behavior][ub]. The compiler
/// will generate code *assuming that you, the programmer, ensure that there will never be
/// undefined behavior*. It is therefore your responsibility to guarantee that every value
/// passed to `transmute` is valid at both types `Src` and `Dst`. Failing to uphold this condition
/// may lead to unexpected and unstable compilation results. This makes `transmute` **incredibly
/// unsafe**. `transmute` should be the absolute last resort.
///
/// Because `transmute` is a by-value operation, alignment of the *transmuted values
/// themselves* is not a concern. As with any other function, the compiler already ensures
/// both `Src` and `Dst` are properly aligned. However, when transmuting values that *point
/// elsewhere* (such as pointers, references, boxes…), the caller has to ensure proper
/// alignment of the pointed-to values.
///
/// The [nomicon](../../nomicon/transmutes.html) has additional documentation.
///
/// [ub]: ../../reference/behavior-considered-undefined.html
///
/// # Transmutation between pointers and integers
///
/// Special care has to be taken when transmuting between pointers and integers, e.g.
/// transmuting between `*const ()` and `usize`.
///
/// Transmuting *pointers to integers* in a `const` context is [undefined behavior][ub], unless
/// the pointer was originally created *from* an integer. (That includes this function
/// specifically, integer-to-pointer casts, and helpers like [`dangling`][crate::ptr::dangling],
/// but also semantically-equivalent conversions such as punning through `repr(C)` union
/// fields.) Any attempt to use the resulting value for integer operations will abort
/// const-evaluation. (And even outside `const`, such transmutation is touching on many
/// unspecified aspects of the Rust memory model and should be avoided. See below for
/// alternatives.)
///
/// Transmuting *integers to pointers* is a largely unspecified operation. It is likely *not*
/// equivalent to an `as` cast. Doing non-zero-sized memory accesses with a pointer constructed
/// this way is currently considered undefined behavior.
///
/// All this also applies when the integer is nested inside an array, tuple, struct, or enum.
/// However, `MaybeUninit<usize>` is not considered an integer type for the purpose of this
/// section. Transmuting `*const ()` to `MaybeUninit<usize>` is fine---but then calling
/// `assume_init()` on that result is considered as completing the pointer-to-integer transmute
/// and thus runs into the issues discussed above.
///
/// In particular, doing a pointer-to-integer-to-pointer roundtrip via `transmute` is *not* a
/// lossless process. If you want to round-trip a pointer through an integer in a way that you
/// can get back the original pointer, you need to use `as` casts, or replace the integer type
/// by `MaybeUninit<$int>` (and never call `assume_init()`). If you are looking for a way to
/// store data of arbitrary type, also use `MaybeUninit<T>` (that will also handle uninitialized
/// memory due to padding). If you specifically need to store something that is "either an
/// integer or a pointer", use `*mut ()`: integers can be converted to pointers and back without
/// any loss (via `as` casts or via `transmute`).
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
/// // Crucially, we `as`-cast to a raw pointer before `transmute`ing to a function pointer.
/// // This avoids an integer-to-pointer `transmute`, which can be problematic.
/// // Transmuting between raw pointers and function pointers (i.e., two pointer types) is fine.
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
///     unsafe { std::mem::transmute::<R<'b>, R<'static>>(r) }
/// }
///
/// unsafe fn shorten_invariant_lifetime<'b, 'c>(r: &'b mut R<'static>)
///                                              -> &'b mut R<'c> {
///     unsafe { std::mem::transmute::<&'b mut R<'static>, &'b mut R<'c>>(r) }
/// }
/// ```
///
/// # Alternatives
///
/// Don't despair: many uses of `transmute` can be achieved through other means.
/// Below are common applications of `transmute` which can be replaced with safer
/// constructs.
///
/// Turning raw bytes (`[u8; SZ]`) into `u32`, `f64`, etc.:
///
/// ```
/// # #![allow(unnecessary_transmutes)]
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
/// ```no_run
/// let ptr = &0;
/// let ptr_num_transmute = unsafe {
///     std::mem::transmute::<&i32, usize>(ptr)
/// };
///
/// // Use an `as` cast instead
/// let ptr_num_cast = ptr as *const i32 as usize;
/// ```
///
/// Note that using `transmute` to turn a pointer to a `usize` is (as noted above) [undefined
/// behavior][ub] in `const` contexts. Also outside of consts, this operation might not behave
/// as expected -- this is touching on many unspecified aspects of the Rust memory model.
/// Depending on what the code is doing, the following alternatives are preferable to
/// pointer-to-integer transmutation:
/// - If the code just wants to store data of arbitrary type in some buffer and needs to pick a
///   type for that buffer, it can use [`MaybeUninit`][crate::mem::MaybeUninit].
/// - If the code actually wants to work on the address the pointer points to, it can use `as`
///   casts or [`ptr.addr()`][pointer::addr].
///
/// Turning a `*mut T` into a `&mut T`:
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
/// Turning a `&mut T` into a `&mut U`:
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
/// Turning a `&str` into a `&[u8]`:
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
/// // It may copy the entire vector into a new one though, but also may not.
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
/// // you a `&mut T` from a `&mut T` or `*mut T`.
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
#[rustc_allowed_through_unstable_modules = "import this function via `std::mem` instead"]
#[rustc_const_stable(feature = "const_transmute", since = "1.56.0")]
#[rustc_diagnostic_item = "transmute"]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn transmute<Src, Dst>(src: Src) -> Dst;

/// Like [`transmute`], but even less checked at compile-time: rather than
/// giving an error for `size_of::<Src>() != size_of::<Dst>()`, it's
/// **Undefined Behavior** at runtime.
///
/// Prefer normal `transmute` where possible, for the extra checking, since
/// both do exactly the same thing at runtime, if they both compile.
///
/// This is not expected to ever be exposed directly to users, rather it
/// may eventually be exposed through some more-constrained API.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn transmute_unchecked<Src, Dst>(src: Src) -> Dst;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn needs_drop<T: ?Sized>() -> bool;

/// Calculates the offset from a pointer.
///
/// This is implemented as an intrinsic to avoid converting to and from an
/// integer, since the conversion would throw away aliasing information.
///
/// This can only be used with `Ptr` as a raw pointer type (`*mut` or `*const`)
/// to a `Sized` pointee and with `Delta` as `usize` or `isize`.  Any other
/// instantiations may arbitrarily misbehave, and that's *not* a compiler bug.
///
/// # Safety
///
/// If the computed offset is non-zero, then both the starting and resulting pointer must be
/// either in bounds or at the end of an allocated object. If either pointer is out
/// of bounds or arithmetic overflow occurs then this operation is undefined behavior.
///
/// The stabilized version of this intrinsic is [`pointer::offset`].
#[must_use = "returns a new pointer rather than modifying its argument"]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn offset<Ptr, Delta>(dst: Ptr, offset: Delta) -> Ptr;

/// Calculates the offset from a pointer, potentially wrapping.
///
/// This is implemented as an intrinsic to avoid converting to and from an
/// integer, since the conversion inhibits certain optimizations.
///
/// # Safety
///
/// Unlike the `offset` intrinsic, this intrinsic does not restrict the
/// resulting pointer to point into or at the end of an allocated
/// object, and it wraps with two's complement arithmetic. The resulting
/// value is not necessarily valid to be used to actually access memory.
///
/// The stabilized version of this intrinsic is [`pointer::wrapping_offset`].
#[must_use = "returns a new pointer rather than modifying its argument"]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;

/// Masks out bits of the pointer according to a mask.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// Consider using [`pointer::mask`] instead.
#[rustc_nounwind]
#[rustc_intrinsic]
pub fn ptr_mask<T>(ptr: *const T, mask: usize) -> *const T;

/// Equivalent to the appropriate `llvm.memcpy.p0i8.0i8.*` intrinsic, with
/// a size of `count` * `size_of::<T>()` and an alignment of
/// `min_align_of::<T>()`
///
/// This intrinsic does not have a stable counterpart.
/// # Safety
///
/// The safety requirements are consistent with [`copy_nonoverlapping`]
/// while the read and write behaviors are volatile,
/// which means it will not be optimized out unless `_count` or `size_of::<T>()` is equal to zero.
///
/// [`copy_nonoverlapping`]: ptr::copy_nonoverlapping
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn volatile_copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: usize);
/// Equivalent to the appropriate `llvm.memmove.p0i8.0i8.*` intrinsic, with
/// a size of `count * size_of::<T>()` and an alignment of
/// `min_align_of::<T>()`
///
/// The volatile parameter is set to `true`, so it will not be optimized out
/// unless size is equal to zero.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn volatile_copy_memory<T>(dst: *mut T, src: *const T, count: usize);
/// Equivalent to the appropriate `llvm.memset.p0i8.*` intrinsic, with a
/// size of `count * size_of::<T>()` and an alignment of
/// `min_align_of::<T>()`.
///
/// This intrinsic does not have a stable counterpart.
/// # Safety
///
/// The safety requirements are consistent with [`write_bytes`] while the write behavior is volatile,
/// which means it will not be optimized out unless `_count` or `size_of::<T>()` is equal to zero.
///
/// [`write_bytes`]: ptr::write_bytes
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn volatile_set_memory<T>(dst: *mut T, val: u8, count: usize);

/// Performs a volatile load from the `src` pointer.
///
/// The stabilized version of this intrinsic is [`core::ptr::read_volatile`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn volatile_load<T>(src: *const T) -> T;
/// Performs a volatile store to the `dst` pointer.
///
/// The stabilized version of this intrinsic is [`core::ptr::write_volatile`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn volatile_store<T>(dst: *mut T, val: T);

/// Performs a volatile load from the `src` pointer
/// The pointer is not required to be aligned.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_diagnostic_item = "intrinsics_unaligned_volatile_load"]
pub unsafe fn unaligned_volatile_load<T>(src: *const T) -> T;
/// Performs a volatile store to the `dst` pointer.
/// The pointer is not required to be aligned.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_diagnostic_item = "intrinsics_unaligned_volatile_store"]
pub unsafe fn unaligned_volatile_store<T>(dst: *mut T, val: T);

/// Returns the square root of an `f16`
///
/// The stabilized version of this intrinsic is
/// [`f16::sqrt`](../../std/primitive.f16.html#method.sqrt)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sqrtf16(x: f16) -> f16;
/// Returns the square root of an `f32`
///
/// The stabilized version of this intrinsic is
/// [`f32::sqrt`](../../std/primitive.f32.html#method.sqrt)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sqrtf32(x: f32) -> f32;
/// Returns the square root of an `f64`
///
/// The stabilized version of this intrinsic is
/// [`f64::sqrt`](../../std/primitive.f64.html#method.sqrt)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sqrtf64(x: f64) -> f64;
/// Returns the square root of an `f128`
///
/// The stabilized version of this intrinsic is
/// [`f128::sqrt`](../../std/primitive.f128.html#method.sqrt)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sqrtf128(x: f128) -> f128;

/// Raises an `f16` to an integer power.
///
/// The stabilized version of this intrinsic is
/// [`f16::powi`](../../std/primitive.f16.html#method.powi)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powif16(a: f16, x: i32) -> f16;
/// Raises an `f32` to an integer power.
///
/// The stabilized version of this intrinsic is
/// [`f32::powi`](../../std/primitive.f32.html#method.powi)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powif32(a: f32, x: i32) -> f32;
/// Raises an `f64` to an integer power.
///
/// The stabilized version of this intrinsic is
/// [`f64::powi`](../../std/primitive.f64.html#method.powi)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powif64(a: f64, x: i32) -> f64;
/// Raises an `f128` to an integer power.
///
/// The stabilized version of this intrinsic is
/// [`f128::powi`](../../std/primitive.f128.html#method.powi)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powif128(a: f128, x: i32) -> f128;

/// Returns the sine of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::sin`](../../std/primitive.f16.html#method.sin)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sinf16(x: f16) -> f16;
/// Returns the sine of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::sin`](../../std/primitive.f32.html#method.sin)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sinf32(x: f32) -> f32;
/// Returns the sine of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::sin`](../../std/primitive.f64.html#method.sin)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sinf64(x: f64) -> f64;
/// Returns the sine of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::sin`](../../std/primitive.f128.html#method.sin)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sinf128(x: f128) -> f128;

/// Returns the cosine of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::cos`](../../std/primitive.f16.html#method.cos)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn cosf16(x: f16) -> f16;
/// Returns the cosine of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::cos`](../../std/primitive.f32.html#method.cos)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn cosf32(x: f32) -> f32;
/// Returns the cosine of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::cos`](../../std/primitive.f64.html#method.cos)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn cosf64(x: f64) -> f64;
/// Returns the cosine of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::cos`](../../std/primitive.f128.html#method.cos)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn cosf128(x: f128) -> f128;

/// Raises an `f16` to an `f16` power.
///
/// The stabilized version of this intrinsic is
/// [`f16::powf`](../../std/primitive.f16.html#method.powf)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powf16(a: f16, x: f16) -> f16;
/// Raises an `f32` to an `f32` power.
///
/// The stabilized version of this intrinsic is
/// [`f32::powf`](../../std/primitive.f32.html#method.powf)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powf32(a: f32, x: f32) -> f32;
/// Raises an `f64` to an `f64` power.
///
/// The stabilized version of this intrinsic is
/// [`f64::powf`](../../std/primitive.f64.html#method.powf)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powf64(a: f64, x: f64) -> f64;
/// Raises an `f128` to an `f128` power.
///
/// The stabilized version of this intrinsic is
/// [`f128::powf`](../../std/primitive.f128.html#method.powf)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn powf128(a: f128, x: f128) -> f128;

/// Returns the exponential of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::exp`](../../std/primitive.f16.html#method.exp)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn expf16(x: f16) -> f16;
/// Returns the exponential of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::exp`](../../std/primitive.f32.html#method.exp)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn expf32(x: f32) -> f32;
/// Returns the exponential of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::exp`](../../std/primitive.f64.html#method.exp)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn expf64(x: f64) -> f64;
/// Returns the exponential of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::exp`](../../std/primitive.f128.html#method.exp)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn expf128(x: f128) -> f128;

/// Returns 2 raised to the power of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::exp2`](../../std/primitive.f16.html#method.exp2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn exp2f16(x: f16) -> f16;
/// Returns 2 raised to the power of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::exp2`](../../std/primitive.f32.html#method.exp2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn exp2f32(x: f32) -> f32;
/// Returns 2 raised to the power of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::exp2`](../../std/primitive.f64.html#method.exp2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn exp2f64(x: f64) -> f64;
/// Returns 2 raised to the power of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::exp2`](../../std/primitive.f128.html#method.exp2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn exp2f128(x: f128) -> f128;

/// Returns the natural logarithm of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::ln`](../../std/primitive.f16.html#method.ln)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn logf16(x: f16) -> f16;
/// Returns the natural logarithm of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::ln`](../../std/primitive.f32.html#method.ln)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn logf32(x: f32) -> f32;
/// Returns the natural logarithm of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::ln`](../../std/primitive.f64.html#method.ln)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn logf64(x: f64) -> f64;
/// Returns the natural logarithm of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::ln`](../../std/primitive.f128.html#method.ln)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn logf128(x: f128) -> f128;

/// Returns the base 10 logarithm of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::log10`](../../std/primitive.f16.html#method.log10)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log10f16(x: f16) -> f16;
/// Returns the base 10 logarithm of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::log10`](../../std/primitive.f32.html#method.log10)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log10f32(x: f32) -> f32;
/// Returns the base 10 logarithm of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::log10`](../../std/primitive.f64.html#method.log10)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log10f64(x: f64) -> f64;
/// Returns the base 10 logarithm of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::log10`](../../std/primitive.f128.html#method.log10)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log10f128(x: f128) -> f128;

/// Returns the base 2 logarithm of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::log2`](../../std/primitive.f16.html#method.log2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log2f16(x: f16) -> f16;
/// Returns the base 2 logarithm of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::log2`](../../std/primitive.f32.html#method.log2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log2f32(x: f32) -> f32;
/// Returns the base 2 logarithm of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::log2`](../../std/primitive.f64.html#method.log2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log2f64(x: f64) -> f64;
/// Returns the base 2 logarithm of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::log2`](../../std/primitive.f128.html#method.log2)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn log2f128(x: f128) -> f128;

/// Returns `a * b + c` for `f16` values.
///
/// The stabilized version of this intrinsic is
/// [`f16::mul_add`](../../std/primitive.f16.html#method.mul_add)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmaf16(a: f16, b: f16, c: f16) -> f16;
/// Returns `a * b + c` for `f32` values.
///
/// The stabilized version of this intrinsic is
/// [`f32::mul_add`](../../std/primitive.f32.html#method.mul_add)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmaf32(a: f32, b: f32, c: f32) -> f32;
/// Returns `a * b + c` for `f64` values.
///
/// The stabilized version of this intrinsic is
/// [`f64::mul_add`](../../std/primitive.f64.html#method.mul_add)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmaf64(a: f64, b: f64, c: f64) -> f64;
/// Returns `a * b + c` for `f128` values.
///
/// The stabilized version of this intrinsic is
/// [`f128::mul_add`](../../std/primitive.f128.html#method.mul_add)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmaf128(a: f128, b: f128, c: f128) -> f128;

/// Returns `a * b + c` for `f16` values, non-deterministically executing
/// either a fused multiply-add or two operations with rounding of the
/// intermediate result.
///
/// The operation is fused if the code generator determines that target
/// instruction set has support for a fused operation, and that the fused
/// operation is more efficient than the equivalent, separate pair of mul
/// and add instructions. It is unspecified whether or not a fused operation
/// is selected, and that may depend on optimization level and context, for
/// example.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmuladdf16(a: f16, b: f16, c: f16) -> f16;
/// Returns `a * b + c` for `f32` values, non-deterministically executing
/// either a fused multiply-add or two operations with rounding of the
/// intermediate result.
///
/// The operation is fused if the code generator determines that target
/// instruction set has support for a fused operation, and that the fused
/// operation is more efficient than the equivalent, separate pair of mul
/// and add instructions. It is unspecified whether or not a fused operation
/// is selected, and that may depend on optimization level and context, for
/// example.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmuladdf32(a: f32, b: f32, c: f32) -> f32;
/// Returns `a * b + c` for `f64` values, non-deterministically executing
/// either a fused multiply-add or two operations with rounding of the
/// intermediate result.
///
/// The operation is fused if the code generator determines that target
/// instruction set has support for a fused operation, and that the fused
/// operation is more efficient than the equivalent, separate pair of mul
/// and add instructions. It is unspecified whether or not a fused operation
/// is selected, and that may depend on optimization level and context, for
/// example.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmuladdf64(a: f64, b: f64, c: f64) -> f64;
/// Returns `a * b + c` for `f128` values, non-deterministically executing
/// either a fused multiply-add or two operations with rounding of the
/// intermediate result.
///
/// The operation is fused if the code generator determines that target
/// instruction set has support for a fused operation, and that the fused
/// operation is more efficient than the equivalent, separate pair of mul
/// and add instructions. It is unspecified whether or not a fused operation
/// is selected, and that may depend on optimization level and context, for
/// example.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmuladdf128(a: f128, b: f128, c: f128) -> f128;

/// Returns the largest integer less than or equal to an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::floor`](../../std/primitive.f16.html#method.floor)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn floorf16(x: f16) -> f16;
/// Returns the largest integer less than or equal to an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::floor`](../../std/primitive.f32.html#method.floor)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn floorf32(x: f32) -> f32;
/// Returns the largest integer less than or equal to an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::floor`](../../std/primitive.f64.html#method.floor)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn floorf64(x: f64) -> f64;
/// Returns the largest integer less than or equal to an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::floor`](../../std/primitive.f128.html#method.floor)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn floorf128(x: f128) -> f128;

/// Returns the smallest integer greater than or equal to an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::ceil`](../../std/primitive.f16.html#method.ceil)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn ceilf16(x: f16) -> f16;
/// Returns the smallest integer greater than or equal to an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::ceil`](../../std/primitive.f32.html#method.ceil)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn ceilf32(x: f32) -> f32;
/// Returns the smallest integer greater than or equal to an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::ceil`](../../std/primitive.f64.html#method.ceil)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn ceilf64(x: f64) -> f64;
/// Returns the smallest integer greater than or equal to an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::ceil`](../../std/primitive.f128.html#method.ceil)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn ceilf128(x: f128) -> f128;

/// Returns the integer part of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::trunc`](../../std/primitive.f16.html#method.trunc)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn truncf16(x: f16) -> f16;
/// Returns the integer part of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::trunc`](../../std/primitive.f32.html#method.trunc)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn truncf32(x: f32) -> f32;
/// Returns the integer part of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::trunc`](../../std/primitive.f64.html#method.trunc)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn truncf64(x: f64) -> f64;
/// Returns the integer part of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::trunc`](../../std/primitive.f128.html#method.trunc)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn truncf128(x: f128) -> f128;

/// Returns the nearest integer to an `f16`. Rounds half-way cases to the number with an even
/// least significant digit.
///
/// The stabilized version of this intrinsic is
/// [`f16::round_ties_even`](../../std/primitive.f16.html#method.round_ties_even)
#[rustc_intrinsic]
#[rustc_nounwind]
pub fn round_ties_even_f16(x: f16) -> f16;

/// Returns the nearest integer to an `f32`. Rounds half-way cases to the number with an even
/// least significant digit.
///
/// The stabilized version of this intrinsic is
/// [`f32::round_ties_even`](../../std/primitive.f32.html#method.round_ties_even)
#[rustc_intrinsic]
#[rustc_nounwind]
pub fn round_ties_even_f32(x: f32) -> f32;

/// Returns the nearest integer to an `f64`. Rounds half-way cases to the number with an even
/// least significant digit.
///
/// The stabilized version of this intrinsic is
/// [`f64::round_ties_even`](../../std/primitive.f64.html#method.round_ties_even)
#[rustc_intrinsic]
#[rustc_nounwind]
pub fn round_ties_even_f64(x: f64) -> f64;

/// Returns the nearest integer to an `f128`. Rounds half-way cases to the number with an even
/// least significant digit.
///
/// The stabilized version of this intrinsic is
/// [`f128::round_ties_even`](../../std/primitive.f128.html#method.round_ties_even)
#[rustc_intrinsic]
#[rustc_nounwind]
pub fn round_ties_even_f128(x: f128) -> f128;

/// Returns the nearest integer to an `f16`. Rounds half-way cases away from zero.
///
/// The stabilized version of this intrinsic is
/// [`f16::round`](../../std/primitive.f16.html#method.round)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn roundf16(x: f16) -> f16;
/// Returns the nearest integer to an `f32`. Rounds half-way cases away from zero.
///
/// The stabilized version of this intrinsic is
/// [`f32::round`](../../std/primitive.f32.html#method.round)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn roundf32(x: f32) -> f32;
/// Returns the nearest integer to an `f64`. Rounds half-way cases away from zero.
///
/// The stabilized version of this intrinsic is
/// [`f64::round`](../../std/primitive.f64.html#method.round)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn roundf64(x: f64) -> f64;
/// Returns the nearest integer to an `f128`. Rounds half-way cases away from zero.
///
/// The stabilized version of this intrinsic is
/// [`f128::round`](../../std/primitive.f128.html#method.round)
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn roundf128(x: f128) -> f128;

/// Float addition that allows optimizations based on algebraic rules.
/// May assume inputs are finite.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fadd_fast<T: Copy>(a: T, b: T) -> T;

/// Float subtraction that allows optimizations based on algebraic rules.
/// May assume inputs are finite.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fsub_fast<T: Copy>(a: T, b: T) -> T;

/// Float multiplication that allows optimizations based on algebraic rules.
/// May assume inputs are finite.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fmul_fast<T: Copy>(a: T, b: T) -> T;

/// Float division that allows optimizations based on algebraic rules.
/// May assume inputs are finite.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn fdiv_fast<T: Copy>(a: T, b: T) -> T;

/// Float remainder that allows optimizations based on algebraic rules.
/// May assume inputs are finite.
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn frem_fast<T: Copy>(a: T, b: T) -> T;

/// Converts with LLVM’s fptoui/fptosi, which may return undef for values out of range
/// (<https://github.com/rust-lang/rust/issues/10184>)
///
/// Stabilized as [`f32::to_int_unchecked`] and [`f64::to_int_unchecked`].
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn float_to_int_unchecked<Float: Copy, Int: Copy>(value: Float) -> Int;

/// Float addition that allows optimizations based on algebraic rules.
///
/// Stabilized as [`f16::algebraic_add`], [`f32::algebraic_add`], [`f64::algebraic_add`] and [`f128::algebraic_add`].
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn fadd_algebraic<T: Copy>(a: T, b: T) -> T;

/// Float subtraction that allows optimizations based on algebraic rules.
///
/// Stabilized as [`f16::algebraic_sub`], [`f32::algebraic_sub`], [`f64::algebraic_sub`] and [`f128::algebraic_sub`].
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn fsub_algebraic<T: Copy>(a: T, b: T) -> T;

/// Float multiplication that allows optimizations based on algebraic rules.
///
/// Stabilized as [`f16::algebraic_mul`], [`f32::algebraic_mul`], [`f64::algebraic_mul`] and [`f128::algebraic_mul`].
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn fmul_algebraic<T: Copy>(a: T, b: T) -> T;

/// Float division that allows optimizations based on algebraic rules.
///
/// Stabilized as [`f16::algebraic_div`], [`f32::algebraic_div`], [`f64::algebraic_div`] and [`f128::algebraic_div`].
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn fdiv_algebraic<T: Copy>(a: T, b: T) -> T;

/// Float remainder that allows optimizations based on algebraic rules.
///
/// Stabilized as [`f16::algebraic_rem`], [`f32::algebraic_rem`], [`f64::algebraic_rem`] and [`f128::algebraic_rem`].
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn frem_algebraic<T: Copy>(a: T, b: T) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn ctpop<T: Copy>(x: T) -> u32;

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
/// # #![allow(internal_features)]
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
/// # #![allow(internal_features)]
///
/// use std::intrinsics::ctlz;
///
/// let x = 0u16;
/// let num_leading = ctlz(x);
/// assert_eq!(num_leading, 16);
/// ```
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn ctlz<T: Copy>(x: T) -> u32;

/// Like `ctlz`, but extra-unsafe as it returns `undef` when
/// given an `x` with value `0`.
///
/// This intrinsic does not have a stable counterpart.
///
/// # Examples
///
/// ```
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
///
/// use std::intrinsics::ctlz_nonzero;
///
/// let x = 0b0001_1100_u8;
/// let num_leading = unsafe { ctlz_nonzero(x) };
/// assert_eq!(num_leading, 3);
/// ```
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn ctlz_nonzero<T: Copy>(x: T) -> u32;

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
/// # #![allow(internal_features)]
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
/// # #![allow(internal_features)]
///
/// use std::intrinsics::cttz;
///
/// let x = 0u16;
/// let num_trailing = cttz(x);
/// assert_eq!(num_trailing, 16);
/// ```
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn cttz<T: Copy>(x: T) -> u32;

/// Like `cttz`, but extra-unsafe as it returns `undef` when
/// given an `x` with value `0`.
///
/// This intrinsic does not have a stable counterpart.
///
/// # Examples
///
/// ```
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
///
/// use std::intrinsics::cttz_nonzero;
///
/// let x = 0b0011_1000_u8;
/// let num_trailing = unsafe { cttz_nonzero(x) };
/// assert_eq!(num_trailing, 3);
/// ```
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn cttz_nonzero<T: Copy>(x: T) -> u32;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn bswap<T: Copy>(x: T) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn bitreverse<T: Copy>(x: T) -> T;

/// Does a three-way comparison between the two arguments,
/// which must be of character or integer (signed or unsigned) type.
///
/// This was originally added because it greatly simplified the MIR in `cmp`
/// implementations, and then LLVM 20 added a backend intrinsic for it too.
///
/// The stabilized version of this intrinsic is [`Ord::cmp`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn three_way_compare<T: Copy>(lhs: T, rhss: T) -> crate::cmp::Ordering;

/// Combine two values which have no bits in common.
///
/// This allows the backend to implement it as `a + b` *or* `a | b`,
/// depending which is easier to implement on a specific target.
///
/// # Safety
///
/// Requires that `(a & b) == 0`, or equivalently that `(a | b) == (a + b)`.
///
/// Otherwise it's immediate UB.
#[rustc_const_unstable(feature = "disjoint_bitor", issue = "135758")]
#[rustc_nounwind]
#[rustc_intrinsic]
#[track_caller]
#[miri::intrinsic_fallback_is_spec] // the fallbacks all `assume` to tell Miri
pub const unsafe fn disjoint_bitor<T: ~const fallback::DisjointBitOr>(a: T, b: T) -> T {
    // SAFETY: same preconditions as this function.
    unsafe { fallback::DisjointBitOr::disjoint_bitor(a, b) }
}

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn add_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn sub_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn mul_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

/// Performs full-width multiplication and addition with a carry:
/// `multiplier * multiplicand + addend + carry`.
///
/// This is possible without any overflow.  For `uN`:
///    MAX * MAX + MAX + MAX
/// => (2ⁿ-1) × (2ⁿ-1) + (2ⁿ-1) + (2ⁿ-1)
/// => (2²ⁿ - 2ⁿ⁺¹ + 1) + (2ⁿ⁺¹ - 2)
/// => 2²ⁿ - 1
///
/// For `iN`, the upper bound is MIN * MIN + MAX + MAX => 2²ⁿ⁻² + 2ⁿ - 2,
/// and the lower bound is MAX * MIN + MIN + MIN => -2²ⁿ⁻² - 2ⁿ + 2ⁿ⁺¹.
///
/// This currently supports unsigned integers *only*, no signed ones.
/// The stabilized versions of this intrinsic are available on integers.
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_const_unstable(feature = "const_carrying_mul_add", issue = "85532")]
#[rustc_nounwind]
#[rustc_intrinsic]
#[miri::intrinsic_fallback_is_spec]
pub const fn carrying_mul_add<T: ~const fallback::CarryingMulAdd<Unsigned = U>, U>(
    multiplier: T,
    multiplicand: T,
    addend: T,
    carry: T,
) -> (U, T) {
    multiplier.carrying_mul_add(multiplicand, addend, carry)
}

/// Performs an exact division, resulting in undefined behavior where
/// `x % y != 0` or `y == 0` or `x == T::MIN && y == -1`
///
/// This intrinsic does not have a stable counterpart.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn exact_div<T: Copy>(x: T, y: T) -> T;

/// Performs an unchecked division, resulting in undefined behavior
/// where `y == 0` or `x == T::MIN && y == -1`
///
/// Safe wrappers for this intrinsic are available on the integer
/// primitives via the `checked_div` method. For example,
/// [`u32::checked_div`]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_div<T: Copy>(x: T, y: T) -> T;
/// Returns the remainder of an unchecked division, resulting in
/// undefined behavior when `y == 0` or `x == T::MIN && y == -1`
///
/// Safe wrappers for this intrinsic are available on the integer
/// primitives via the `checked_rem` method. For example,
/// [`u32::checked_rem`]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_rem<T: Copy>(x: T, y: T) -> T;

/// Performs an unchecked left shift, resulting in undefined behavior when
/// `y < 0` or `y >= N`, where N is the width of T in bits.
///
/// Safe wrappers for this intrinsic are available on the integer
/// primitives via the `checked_shl` method. For example,
/// [`u32::checked_shl`]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_shl<T: Copy, U: Copy>(x: T, y: U) -> T;
/// Performs an unchecked right shift, resulting in undefined behavior when
/// `y < 0` or `y >= N`, where N is the width of T in bits.
///
/// Safe wrappers for this intrinsic are available on the integer
/// primitives via the `checked_shr` method. For example,
/// [`u32::checked_shr`]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_shr<T: Copy, U: Copy>(x: T, y: U) -> T;

/// Returns the result of an unchecked addition, resulting in
/// undefined behavior when `x + y > T::MAX` or `x + y < T::MIN`.
///
/// The stable counterpart of this intrinsic is `unchecked_add` on the various
/// integer types, such as [`u16::unchecked_add`] and [`i64::unchecked_add`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_add<T: Copy>(x: T, y: T) -> T;

/// Returns the result of an unchecked subtraction, resulting in
/// undefined behavior when `x - y > T::MAX` or `x - y < T::MIN`.
///
/// The stable counterpart of this intrinsic is `unchecked_sub` on the various
/// integer types, such as [`u16::unchecked_sub`] and [`i64::unchecked_sub`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_sub<T: Copy>(x: T, y: T) -> T;

/// Returns the result of an unchecked multiplication, resulting in
/// undefined behavior when `x * y > T::MAX` or `x * y < T::MIN`.
///
/// The stable counterpart of this intrinsic is `unchecked_mul` on the various
/// integer types, such as [`u16::unchecked_mul`] and [`i64::unchecked_mul`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn unchecked_mul<T: Copy>(x: T, y: T) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn rotate_left<T: Copy>(x: T, shift: u32) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn rotate_right<T: Copy>(x: T, shift: u32) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn wrapping_add<T: Copy>(a: T, b: T) -> T;
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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn wrapping_sub<T: Copy>(a: T, b: T) -> T;
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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn wrapping_mul<T: Copy>(a: T, b: T) -> T;

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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn saturating_add<T: Copy>(a: T, b: T) -> T;
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
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn saturating_sub<T: Copy>(a: T, b: T) -> T;

/// This is an implementation detail of [`crate::ptr::read`] and should
/// not be used anywhere else.  See its comments for why this exists.
///
/// This intrinsic can *only* be called where the pointer is a local without
/// projections (`read_via_copy(ptr)`, not `read_via_copy(*ptr)`) so that it
/// trivially obeys runtime-MIR rules about derefs in operands.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn read_via_copy<T>(ptr: *const T) -> T;

/// This is an implementation detail of [`crate::ptr::write`] and should
/// not be used anywhere else.  See its comments for why this exists.
///
/// This intrinsic can *only* be called where the pointer is a local without
/// projections (`write_via_move(ptr, x)`, not `write_via_move(*ptr, x)`) so
/// that it trivially obeys runtime-MIR rules about derefs in operands.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn write_via_move<T>(ptr: *mut T, value: T);

/// Returns the value of the discriminant for the variant in 'v';
/// if `T` has no discriminant, returns `0`.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is [`core::mem::discriminant`].
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;

/// Rust's "try catch" construct for unwinding. Invokes the function pointer `try_fn` with the
/// data pointer `data`, and calls `catch_fn` if unwinding occurs while `try_fn` runs.
/// Returns `1` if unwinding occurred and `catch_fn` was called; returns `0` otherwise.
///
/// `catch_fn` must not unwind.
///
/// The third argument is a function called if an unwind occurs (both Rust `panic` and foreign
/// unwinds). This function takes the data pointer and a pointer to the target- and
/// runtime-specific exception object that was caught.
///
/// Note that in the case of a foreign unwinding operation, the exception object data may not be
/// safely usable from Rust, and should not be directly exposed via the standard library. To
/// prevent unsafe access, the library implementation may either abort the process or present an
/// opaque error type to the user.
///
/// For more information, see the compiler's source, as well as the documentation for the stable
/// version of this intrinsic, `std::panic::catch_unwind`.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn catch_unwind(
    _try_fn: fn(*mut u8),
    _data: *mut u8,
    _catch_fn: fn(*mut u8, *mut u8),
) -> i32;

/// Emits a `nontemporal` store, which gives a hint to the CPU that the data should not be held
/// in cache. Except for performance, this is fully equivalent to `ptr.write(val)`.
///
/// Not all architectures provide such an operation. For instance, x86 does not: while `MOVNT`
/// exists, that operation is *not* equivalent to `ptr.write(val)` (`MOVNT` writes can be reordered
/// in ways that are not allowed for regular writes).
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn nontemporal_store<T>(ptr: *mut T, val: T);

/// See documentation of `<*const T>::offset_from` for details.
#[rustc_intrinsic_const_stable_indirect]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn ptr_offset_from<T>(ptr: *const T, base: *const T) -> isize;

/// See documentation of `<*const T>::offset_from_unsigned` for details.
#[rustc_nounwind]
#[rustc_intrinsic]
#[rustc_intrinsic_const_stable_indirect]
pub const unsafe fn ptr_offset_from_unsigned<T>(ptr: *const T, base: *const T) -> usize;

/// See documentation of `<*const T>::guaranteed_eq` for details.
/// Returns `2` if the result is unknown.
/// Returns `1` if the pointers are guaranteed equal.
/// Returns `0` if the pointers are guaranteed inequal.
#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_do_not_const_check]
#[inline]
#[miri::intrinsic_fallback_is_spec]
pub const fn ptr_guaranteed_cmp<T>(ptr: *const T, other: *const T) -> u8 {
    (ptr == other) as u8
}

/// Determines whether the raw bytes of the two values are equal.
///
/// This is particularly handy for arrays, since it allows things like just
/// comparing `i96`s instead of forcing `alloca`s for `[6 x i16]`.
///
/// Above some backend-decided threshold this will emit calls to `memcmp`,
/// like slice equality does, instead of causing massive code size.
///
/// Since this works by comparing the underlying bytes, the actual `T` is
/// not particularly important.  It will be used for its size and alignment,
/// but any validity restrictions will be ignored, not enforced.
///
/// # Safety
///
/// It's UB to call this if any of the *bytes* in `*a` or `*b` are uninitialized.
/// Note that this is a stricter criterion than just the *values* being
/// fully-initialized: if `T` has padding, it's UB to call this intrinsic.
///
/// At compile-time, it is furthermore UB to call this if any of the bytes
/// in `*a` or `*b` have provenance.
///
/// (The implementation is allowed to branch on the results of comparisons,
/// which is UB if any of their inputs are `undef`.)
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn raw_eq<T>(a: &T, b: &T) -> bool;

/// Lexicographically compare `[left, left + bytes)` and `[right, right + bytes)`
/// as unsigned bytes, returning negative if `left` is less, zero if all the
/// bytes match, or positive if `left` is greater.
///
/// This underlies things like `<[u8]>::cmp`, and will usually lower to `memcmp`.
///
/// # Safety
///
/// `left` and `right` must each be [valid] for reads of `bytes` bytes.
///
/// Note that this applies to the whole range, not just until the first byte
/// that differs.  That allows optimizations that can read in large chunks.
///
/// [valid]: crate::ptr#safety
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn compare_bytes(left: *const u8, right: *const u8, bytes: usize) -> i32;

/// See documentation of [`std::hint::black_box`] for details.
///
/// [`std::hint::black_box`]: crate::hint::black_box
#[rustc_nounwind]
#[rustc_intrinsic]
#[rustc_intrinsic_const_stable_indirect]
pub const fn black_box<T>(dummy: T) -> T;

/// Selects which function to call depending on the context.
///
/// If this function is evaluated at compile-time, then a call to this
/// intrinsic will be replaced with a call to `called_in_const`. It gets
/// replaced with a call to `called_at_rt` otherwise.
///
/// This function is safe to call, but note the stability concerns below.
///
/// # Type Requirements
///
/// The two functions must be both function items. They cannot be function
/// pointers or closures. The first function must be a `const fn`.
///
/// `arg` will be the tupled arguments that will be passed to either one of
/// the two functions, therefore, both functions must accept the same type of
/// arguments. Both functions must return RET.
///
/// # Stability concerns
///
/// Rust has not yet decided that `const fn` are allowed to tell whether
/// they run at compile-time or at runtime. Therefore, when using this
/// intrinsic anywhere that can be reached from stable, it is crucial that
/// the end-to-end behavior of the stable `const fn` is the same for both
/// modes of execution. (Here, Undefined Behavior is considered "the same"
/// as any other behavior, so if the function exhibits UB at runtime then
/// it may do whatever it wants at compile-time.)
///
/// Here is an example of how this could cause a problem:
/// ```no_run
/// #![feature(const_eval_select)]
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
/// use std::intrinsics::const_eval_select;
///
/// // Standard library
/// pub const fn inconsistent() -> i32 {
///     fn runtime() -> i32 { 1 }
///     const fn compiletime() -> i32 { 2 }
///
///     // ⚠ This code violates the required equivalence of `compiletime`
///     // and `runtime`.
///     const_eval_select((), compiletime, runtime)
/// }
///
/// // User Crate
/// const X: i32 = inconsistent();
/// let x = inconsistent();
/// assert_eq!(x, X);
/// ```
///
/// Currently such an assertion would always succeed; until Rust decides
/// otherwise, that principle should not be violated.
#[rustc_const_unstable(feature = "const_eval_select", issue = "124625")]
#[rustc_intrinsic]
pub const fn const_eval_select<ARG: Tuple, F, G, RET>(
    _arg: ARG,
    _called_in_const: F,
    _called_at_rt: G,
) -> RET
where
    G: FnOnce<ARG, Output = RET>,
    F: FnOnce<ARG, Output = RET>;

/// A macro to make it easier to invoke const_eval_select. Use as follows:
/// ```rust,ignore (just a macro example)
/// const_eval_select!(
///     @capture { arg1: i32 = some_expr, arg2: T = other_expr } -> U:
///     if const #[attributes_for_const_arm] {
///         // Compile-time code goes here.
///     } else #[attributes_for_runtime_arm] {
///         // Run-time code goes here.
///     }
/// )
/// ```
/// The `@capture` block declares which surrounding variables / expressions can be
/// used inside the `if const`.
/// Note that the two arms of this `if` really each become their own function, which is why the
/// macro supports setting attributes for those functions. The runtime function is always
/// markes as `#[inline]`.
///
/// See [`const_eval_select()`] for the rules and requirements around that intrinsic.
pub(crate) macro const_eval_select {
    (
        @capture$([$($binders:tt)*])? { $($arg:ident : $ty:ty = $val:expr),* $(,)? } $( -> $ret:ty )? :
        if const
            $(#[$compiletime_attr:meta])* $compiletime:block
        else
            $(#[$runtime_attr:meta])* $runtime:block
    ) => {
        // Use the `noinline` arm, after adding explicit `inline` attributes
        $crate::intrinsics::const_eval_select!(
            @capture$([$($binders)*])? { $($arg : $ty = $val),* } $(-> $ret)? :
            #[noinline]
            if const
                #[inline] // prevent codegen on this function
                $(#[$compiletime_attr])*
                $compiletime
            else
                #[inline] // avoid the overhead of an extra fn call
                $(#[$runtime_attr])*
                $runtime
        )
    },
    // With a leading #[noinline], we don't add inline attributes
    (
        @capture$([$($binders:tt)*])? { $($arg:ident : $ty:ty = $val:expr),* $(,)? } $( -> $ret:ty )? :
        #[noinline]
        if const
            $(#[$compiletime_attr:meta])* $compiletime:block
        else
            $(#[$runtime_attr:meta])* $runtime:block
    ) => {{
        $(#[$runtime_attr])*
        fn runtime$(<$($binders)*>)?($($arg: $ty),*) $( -> $ret )? {
            $runtime
        }

        $(#[$compiletime_attr])*
        const fn compiletime$(<$($binders)*>)?($($arg: $ty),*) $( -> $ret )? {
            // Don't warn if one of the arguments is unused.
            $(let _ = $arg;)*

            $compiletime
        }

        const_eval_select(($($val,)*), compiletime, runtime)
    }},
    // We support leaving away the `val` expressions for *all* arguments
    // (but not for *some* arguments, that's too tricky).
    (
        @capture$([$($binders:tt)*])? { $($arg:ident : $ty:ty),* $(,)? } $( -> $ret:ty )? :
        if const
            $(#[$compiletime_attr:meta])* $compiletime:block
        else
            $(#[$runtime_attr:meta])* $runtime:block
    ) => {
        $crate::intrinsics::const_eval_select!(
            @capture$([$($binders)*])? { $($arg : $ty = $arg),* } $(-> $ret)? :
            if const
                $(#[$compiletime_attr])* $compiletime
            else
                $(#[$runtime_attr])* $runtime
        )
    },
}

/// Returns whether the argument's value is statically known at
/// compile-time.
///
/// This is useful when there is a way of writing the code that will
/// be *faster* when some variables have known values, but *slower*
/// in the general case: an `if is_val_statically_known(var)` can be used
/// to select between these two variants. The `if` will be optimized away
/// and only the desired branch remains.
///
/// Formally speaking, this function non-deterministically returns `true`
/// or `false`, and the caller has to ensure sound behavior for both cases.
/// In other words, the following code has *Undefined Behavior*:
///
/// ```no_run
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
/// use std::hint::unreachable_unchecked;
/// use std::intrinsics::is_val_statically_known;
///
/// if !is_val_statically_known(0) { unsafe { unreachable_unchecked(); } }
/// ```
///
/// This also means that the following code's behavior is unspecified; it
/// may panic, or it may not:
///
/// ```no_run
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
/// use std::intrinsics::is_val_statically_known;
///
/// assert_eq!(is_val_statically_known(0), is_val_statically_known(0));
/// ```
///
/// Unsafe code may not rely on `is_val_statically_known` returning any
/// particular value, ever. However, the compiler will generally make it
/// return `true` only if the value of the argument is actually known.
///
/// # Stability concerns
///
/// While it is safe to call, this intrinsic may behave differently in
/// a `const` context than otherwise. See the [`const_eval_select()`]
/// documentation for an explanation of the issues this can cause. Unlike
/// `const_eval_select`, this intrinsic isn't guaranteed to behave
/// deterministically even in a `const` context.
///
/// # Type Requirements
///
/// `T` must be either a `bool`, a `char`, a primitive numeric type (e.g. `f32`,
/// but not `NonZeroISize`), or any thin pointer (e.g. `*mut String`).
/// Any other argument types *may* cause a compiler error.
///
/// ## Pointers
///
/// When the input is a pointer, only the pointer itself is
/// ever considered. The pointee has no effect. Currently, these functions
/// behave identically:
///
/// ```
/// #![feature(core_intrinsics)]
/// # #![allow(internal_features)]
/// use std::intrinsics::is_val_statically_known;
///
/// fn foo(x: &i32) -> bool {
///     is_val_statically_known(x)
/// }
///
/// fn bar(x: &i32) -> bool {
///     is_val_statically_known(
///         (x as *const i32).addr()
///     )
/// }
/// # _ = foo(&5_i32);
/// # _ = bar(&5_i32);
/// ```
#[rustc_const_stable_indirect]
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const fn is_val_statically_known<T: Copy>(_arg: T) -> bool {
    false
}

/// Non-overlapping *typed* swap of a single value.
///
/// The codegen backends will replace this with a better implementation when
/// `T` is a simple type that can be loaded and stored as an immediate.
///
/// The stabilized form of this intrinsic is [`crate::mem::swap`].
///
/// # Safety
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for both reads and writes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// * The region of memory beginning at `x` must *not* overlap with the region of memory
///   beginning at `y`.
///
/// * The memory pointed by `x` and `y` must both contain values of type `T`.
///
/// [valid]: crate::ptr#safety
#[rustc_nounwind]
#[inline]
#[rustc_intrinsic]
#[rustc_intrinsic_const_stable_indirect]
pub const unsafe fn typed_swap_nonoverlapping<T>(x: *mut T, y: *mut T) {
    // SAFETY: The caller provided single non-overlapping items behind
    // pointers, so swapping them with `count: 1` is fine.
    unsafe { ptr::swap_nonoverlapping(x, y, 1) };
}

/// Returns whether we should perform some UB-checking at runtime. This eventually evaluates to
/// `cfg!(ub_checks)`, but behaves different from `cfg!` when mixing crates built with different
/// flags: if the crate has UB checks enabled or carries the `#[rustc_preserve_ub_checks]`
/// attribute, evaluation is delayed until monomorphization (or until the call gets inlined into
/// a crate that does not delay evaluation further); otherwise it can happen any time.
///
/// The common case here is a user program built with ub_checks linked against the distributed
/// sysroot which is built without ub_checks but with `#[rustc_preserve_ub_checks]`.
/// For code that gets monomorphized in the user crate (i.e., generic functions and functions with
/// `#[inline]`), gating assertions on `ub_checks()` rather than `cfg!(ub_checks)` means that
/// assertions are enabled whenever the *user crate* has UB checks enabled. However, if the
/// user has UB checks disabled, the checks will still get optimized out. This intrinsic is
/// primarily used by [`crate::ub_checks::assert_unsafe_precondition`].
#[rustc_intrinsic_const_stable_indirect] // just for UB checks
#[inline(always)]
#[rustc_intrinsic]
pub const fn ub_checks() -> bool {
    cfg!(ub_checks)
}

/// Allocates a block of memory at compile time.
/// At runtime, just returns a null pointer.
///
/// # Safety
///
/// - The `align` argument must be a power of two.
///    - At compile time, a compile error occurs if this constraint is violated.
///    - At runtime, it is not checked.
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
#[rustc_nounwind]
#[rustc_intrinsic]
#[miri::intrinsic_fallback_is_spec]
pub const unsafe fn const_allocate(_size: usize, _align: usize) -> *mut u8 {
    // const eval overrides this function, but runtime code for now just returns null pointers.
    // See <https://github.com/rust-lang/rust/issues/93935>.
    crate::ptr::null_mut()
}

/// Deallocates a memory which allocated by `intrinsics::const_allocate` at compile time.
/// At runtime, does nothing.
///
/// # Safety
///
/// - The `align` argument must be a power of two.
///    - At compile time, a compile error occurs if this constraint is violated.
///    - At runtime, it is not checked.
/// - If the `ptr` is created in an another const, this intrinsic doesn't deallocate it.
/// - If the `ptr` is pointing to a local variable, this intrinsic doesn't deallocate it.
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_nounwind]
#[rustc_intrinsic]
#[miri::intrinsic_fallback_is_spec]
pub const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {
    // Runtime NOP
}

/// Returns whether we should perform contract-checking at runtime.
///
/// This is meant to be similar to the ub_checks intrinsic, in terms
/// of not prematurely commiting at compile-time to whether contract
/// checking is turned on, so that we can specify contracts in libstd
/// and let an end user opt into turning them on.
#[rustc_const_unstable(feature = "contracts_internals", issue = "128044" /* compiler-team#759 */)]
#[unstable(feature = "contracts_internals", issue = "128044" /* compiler-team#759 */)]
#[inline(always)]
#[rustc_intrinsic]
pub const fn contract_checks() -> bool {
    // FIXME: should this be `false` or `cfg!(contract_checks)`?

    // cfg!(contract_checks)
    false
}

/// Check if the pre-condition `cond` has been met.
///
/// By default, if `contract_checks` is enabled, this will panic with no unwind if the condition
/// returns false.
///
/// Note that this function is a no-op during constant evaluation.
#[unstable(feature = "contracts_internals", issue = "128044")]
// Calls to this function get inserted by an AST expansion pass, which uses the equivalent of
// `#[allow_internal_unstable]` to allow using `contracts_internals` functions. Const-checking
// doesn't honor `#[allow_internal_unstable]`, so for the const feature gate we use the user-facing
// `contracts` feature rather than the perma-unstable `contracts_internals`
#[rustc_const_unstable(feature = "contracts", issue = "128044")]
#[lang = "contract_check_requires"]
#[rustc_intrinsic]
pub const fn contract_check_requires<C: Fn() -> bool + Copy>(cond: C) {
    const_eval_select!(
        @capture[C: Fn() -> bool + Copy] { cond: C } :
        if const {
                // Do nothing
        } else {
            if contract_checks() && !cond() {
                // Emit no unwind panic in case this was a safety requirement.
                crate::panicking::panic_nounwind("failed requires check");
            }
        }
    )
}

/// Check if the post-condition `cond` has been met.
///
/// By default, if `contract_checks` is enabled, this will panic with no unwind if the condition
/// returns false.
///
/// Note that this function is a no-op during constant evaluation.
#[unstable(feature = "contracts_internals", issue = "128044")]
// Similar to `contract_check_requires`, we need to use the user-facing
// `contracts` feature rather than the perma-unstable `contracts_internals`.
// Const-checking doesn't honor allow_internal_unstable logic used by contract expansion.
#[rustc_const_unstable(feature = "contracts", issue = "128044")]
#[lang = "contract_check_ensures"]
#[rustc_intrinsic]
pub const fn contract_check_ensures<C: Fn(&Ret) -> bool + Copy, Ret>(cond: C, ret: Ret) -> Ret {
    const_eval_select!(
        @capture[C: Fn(&Ret) -> bool + Copy, Ret] { cond: C, ret: Ret } -> Ret :
        if const {
            // Do nothing
            ret
        } else {
            if contract_checks() && !cond(&ret) {
                // Emit no unwind panic in case this was a safety requirement.
                crate::panicking::panic_nounwind("failed ensures check");
            }
            ret
        }
    )
}

/// The intrinsic will return the size stored in that vtable.
///
/// # Safety
///
/// `ptr` must point to a vtable.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub unsafe fn vtable_size(ptr: *const ()) -> usize;

/// The intrinsic will return the alignment stored in that vtable.
///
/// # Safety
///
/// `ptr` must point to a vtable.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub unsafe fn vtable_align(ptr: *const ()) -> usize;

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
/// The stabilized version of this intrinsic is [`size_of`].
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn size_of<T>() -> usize;

/// The minimum alignment of a type.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is [`align_of`].
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn min_align_of<T>() -> usize;

/// The preferred alignment of a type.
///
/// This intrinsic does not have a stable counterpart.
/// It's "tracking issue" is [#91971](https://github.com/rust-lang/rust/issues/91971).
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const unsafe fn pref_align_of<T>() -> usize;

/// Returns the number of variants of the type `T` cast to a `usize`;
/// if `T` has no variants, returns `0`. Uninhabited variants will be counted.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The to-be-stabilized version of this intrinsic is [`crate::mem::variant_count`].
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const fn variant_count<T>() -> usize;

/// The size of the referenced value in bytes.
///
/// The stabilized version of this intrinsic is [`size_of_val`].
///
/// # Safety
///
/// See [`crate::mem::size_of_val_raw`] for safety conditions.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
#[rustc_intrinsic_const_stable_indirect]
pub const unsafe fn size_of_val<T: ?Sized>(ptr: *const T) -> usize;

/// The required alignment of the referenced value.
///
/// The stabilized version of this intrinsic is [`align_of_val`].
///
/// # Safety
///
/// See [`crate::mem::align_of_val_raw`] for safety conditions.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
#[rustc_intrinsic_const_stable_indirect]
pub const unsafe fn min_align_of_val<T: ?Sized>(ptr: *const T) -> usize;

/// Gets a static string slice containing the name of a type.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is [`core::any::type_name`].
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const fn type_name<T: ?Sized>() -> &'static str;

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
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic]
pub const fn type_id<T: ?Sized + 'static>() -> u128;

/// Lowers in MIR to `Rvalue::Aggregate` with `AggregateKind::RawPtr`.
///
/// This is used to implement functions like `slice::from_raw_parts_mut` and
/// `ptr::from_raw_parts` in a way compatible with the compiler being able to
/// change the possible layouts of pointers.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn aggregate_raw_ptr<P: AggregateRawPtr<D, Metadata = M>, D, M>(data: D, meta: M) -> P;

#[unstable(feature = "core_intrinsics", issue = "none")]
pub trait AggregateRawPtr<D> {
    type Metadata: Copy;
}
impl<P: ?Sized, T: ptr::Thin> AggregateRawPtr<*const T> for *const P {
    type Metadata = <P as ptr::Pointee>::Metadata;
}
impl<P: ?Sized, T: ptr::Thin> AggregateRawPtr<*mut T> for *mut P {
    type Metadata = <P as ptr::Pointee>::Metadata;
}

/// Lowers in MIR to `Rvalue::UnaryOp` with `UnOp::PtrMetadata`.
///
/// This is used to implement functions like `ptr::metadata`.
#[rustc_nounwind]
#[unstable(feature = "core_intrinsics", issue = "none")]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn ptr_metadata<P: ptr::Pointee<Metadata = M> + ?Sized, M>(ptr: *const P) -> M;

/// This is an accidentally-stable alias to [`ptr::copy_nonoverlapping`]; use that instead.
// Note (intentionally not in the doc comment): `ptr::copy_nonoverlapping` adds some extra
// debug assertions; if you are writing compiler tests or code inside the standard library
// that wants to avoid those debug assertions, directly call this intrinsic instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_allowed_through_unstable_modules = "import this function via `std::ptr` instead"]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);

/// This is an accidentally-stable alias to [`ptr::copy`]; use that instead.
// Note (intentionally not in the doc comment): `ptr::copy` adds some extra
// debug assertions; if you are writing compiler tests or code inside the standard library
// that wants to avoid those debug assertions, directly call this intrinsic instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_allowed_through_unstable_modules = "import this function via `std::ptr` instead"]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize);

/// This is an accidentally-stable alias to [`ptr::write_bytes`]; use that instead.
// Note (intentionally not in the doc comment): `ptr::write_bytes` adds some extra
// debug assertions; if you are writing compiler tests or code inside the standard library
// that wants to avoid those debug assertions, directly call this intrinsic instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_allowed_through_unstable_modules = "import this function via `std::ptr` instead"]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.83.0")]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn write_bytes<T>(dst: *mut T, val: u8, count: usize);

/// Returns the minimum (IEEE 754-2008 minNum) of two `f16` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f16::min`]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn minnumf16(x: f16, y: f16) -> f16;

/// Returns the minimum (IEEE 754-2008 minNum) of two `f32` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f32::min`]
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn minnumf32(x: f32, y: f32) -> f32;

/// Returns the minimum (IEEE 754-2008 minNum) of two `f64` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f64::min`]
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn minnumf64(x: f64, y: f64) -> f64;

/// Returns the minimum (IEEE 754-2008 minNum) of two `f128` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f128::min`]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn minnumf128(x: f128, y: f128) -> f128;

/// Returns the minimum (IEEE 754-2019 minimum) of two `f16` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn minimumf16(x: f16, y: f16) -> f16 {
    if x < y {
        x
    } else if y < x {
        y
    } else if x == y {
        if x.is_sign_negative() && y.is_sign_positive() { x } else { y }
    } else {
        // At least one input is NaN. Use `+` to perform NaN propagation and quieting.
        x + y
    }
}

/// Returns the minimum (IEEE 754-2019 minimum) of two `f32` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn minimumf32(x: f32, y: f32) -> f32 {
    if x < y {
        x
    } else if y < x {
        y
    } else if x == y {
        if x.is_sign_negative() && y.is_sign_positive() { x } else { y }
    } else {
        // At least one input is NaN. Use `+` to perform NaN propagation and quieting.
        x + y
    }
}

/// Returns the minimum (IEEE 754-2019 minimum) of two `f64` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn minimumf64(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else if y < x {
        y
    } else if x == y {
        if x.is_sign_negative() && y.is_sign_positive() { x } else { y }
    } else {
        // At least one input is NaN. Use `+` to perform NaN propagation and quieting.
        x + y
    }
}

/// Returns the minimum (IEEE 754-2019 minimum) of two `f128` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn minimumf128(x: f128, y: f128) -> f128 {
    if x < y {
        x
    } else if y < x {
        y
    } else if x == y {
        if x.is_sign_negative() && y.is_sign_positive() { x } else { y }
    } else {
        // At least one input is NaN. Use `+` to perform NaN propagation and quieting.
        x + y
    }
}

/// Returns the maximum (IEEE 754-2008 maxNum) of two `f16` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f16::max`]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn maxnumf16(x: f16, y: f16) -> f16;

/// Returns the maximum (IEEE 754-2008 maxNum) of two `f32` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f32::max`]
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn maxnumf32(x: f32, y: f32) -> f32;

/// Returns the maximum (IEEE 754-2008 maxNum) of two `f64` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f64::max`]
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const fn maxnumf64(x: f64, y: f64) -> f64;

/// Returns the maximum (IEEE 754-2008 maxNum) of two `f128` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
///
/// The stabilized version of this intrinsic is
/// [`f128::max`]
#[rustc_nounwind]
#[rustc_intrinsic]
pub const fn maxnumf128(x: f128, y: f128) -> f128;

/// Returns the maximum (IEEE 754-2019 maximum) of two `f16` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn maximumf16(x: f16, y: f16) -> f16 {
    if x > y {
        x
    } else if y > x {
        y
    } else if x == y {
        if x.is_sign_positive() && y.is_sign_negative() { x } else { y }
    } else {
        x + y
    }
}

/// Returns the maximum (IEEE 754-2019 maximum) of two `f32` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn maximumf32(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else if y > x {
        y
    } else if x == y {
        if x.is_sign_positive() && y.is_sign_negative() { x } else { y }
    } else {
        x + y
    }
}

/// Returns the maximum (IEEE 754-2019 maximum) of two `f64` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn maximumf64(x: f64, y: f64) -> f64 {
    if x > y {
        x
    } else if y > x {
        y
    } else if x == y {
        if x.is_sign_positive() && y.is_sign_negative() { x } else { y }
    } else {
        x + y
    }
}

/// Returns the maximum (IEEE 754-2019 maximum) of two `f128` values.
///
/// Note that, unlike most intrinsics, this is safe to call;
/// it does not require an `unsafe` block.
/// Therefore, implementations must not require the user to uphold
/// any safety invariants.
#[rustc_nounwind]
#[cfg_attr(not(bootstrap), rustc_intrinsic)]
pub const fn maximumf128(x: f128, y: f128) -> f128 {
    if x > y {
        x
    } else if y > x {
        y
    } else if x == y {
        if x.is_sign_positive() && y.is_sign_negative() { x } else { y }
    } else {
        x + y
    }
}

/// Returns the absolute value of an `f16`.
///
/// The stabilized version of this intrinsic is
/// [`f16::abs`](../../std/primitive.f16.html#method.abs)
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn fabsf16(x: f16) -> f16;

/// Returns the absolute value of an `f32`.
///
/// The stabilized version of this intrinsic is
/// [`f32::abs`](../../std/primitive.f32.html#method.abs)
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const unsafe fn fabsf32(x: f32) -> f32;

/// Returns the absolute value of an `f64`.
///
/// The stabilized version of this intrinsic is
/// [`f64::abs`](../../std/primitive.f64.html#method.abs)
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const unsafe fn fabsf64(x: f64) -> f64;

/// Returns the absolute value of an `f128`.
///
/// The stabilized version of this intrinsic is
/// [`f128::abs`](../../std/primitive.f128.html#method.abs)
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn fabsf128(x: f128) -> f128;

/// Copies the sign from `y` to `x` for `f16` values.
///
/// The stabilized version of this intrinsic is
/// [`f16::copysign`](../../std/primitive.f16.html#method.copysign)
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn copysignf16(x: f16, y: f16) -> f16;

/// Copies the sign from `y` to `x` for `f32` values.
///
/// The stabilized version of this intrinsic is
/// [`f32::copysign`](../../std/primitive.f32.html#method.copysign)
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const unsafe fn copysignf32(x: f32, y: f32) -> f32;
/// Copies the sign from `y` to `x` for `f64` values.
///
/// The stabilized version of this intrinsic is
/// [`f64::copysign`](../../std/primitive.f64.html#method.copysign)
#[rustc_nounwind]
#[rustc_intrinsic_const_stable_indirect]
#[rustc_intrinsic]
pub const unsafe fn copysignf64(x: f64, y: f64) -> f64;

/// Copies the sign from `y` to `x` for `f128` values.
///
/// The stabilized version of this intrinsic is
/// [`f128::copysign`](../../std/primitive.f128.html#method.copysign)
#[rustc_nounwind]
#[rustc_intrinsic]
pub const unsafe fn copysignf128(x: f128, y: f128) -> f128;

/// Inform Miri that a given pointer definitely has a certain alignment.
#[cfg(miri)]
#[rustc_allow_const_fn_unstable(const_eval_select)]
pub(crate) const fn miri_promise_symbolic_alignment(ptr: *const (), align: usize) {
    unsafe extern "Rust" {
        /// Miri-provided extern function to promise that a given pointer is properly aligned for
        /// "symbolic" alignment checks. Will fail if the pointer is not actually aligned or `align` is
        /// not a power of two. Has no effect when alignment checks are concrete (which is the default).
        fn miri_promise_symbolic_alignment(ptr: *const (), align: usize);
    }

    const_eval_select!(
        @capture { ptr: *const (), align: usize}:
        if const {
            // Do nothing.
        } else {
            // SAFETY: this call is always safe.
            unsafe {
                miri_promise_symbolic_alignment(ptr, align);
            }
        }
    )
}
