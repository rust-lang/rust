//! CoAlloction-specific types that only apply in heap-based applications (hence not a part of
//!  [::core]).
//!
//! Types here have names with `CoAlloc` prefix. Yes, when using a q ualified path (like
//! ::alloc::co_alloc::CoAllocPref), that involves "stuttering", which is not recommended.
//!
//! However, as per Rust Book the common practice is to import type names fully and access them just
//! with their name (except for cases of conflict). And we don't want the type names any shorter
//! (such `Pref`), because thoe would be vague/confusing.

/// `CoAllocPref` values indicate a type's preference for coallocation (in either user space, or
/// `std` space). Used as a `const` generic parameter type (usually called `CO_ALLOC_PREF`).
///
/// The actual value may be overriden by the allocator. See also `CoAllocMetaNumSlotsPref` and
/// `co_alloc_pref` macro .
///
/// This type  WILL CHANGE (once ``#![feature(generic_const_exprs)]` and
/// `#![feature(adt_const_params)]` are stable) to a dedicated struct/enum. Hence:
/// - DO NOT construct instances, but use `co_alloc_pref` macro together with constants
/// `CO_ALLOC_PREF_META_YES` and `CO_ALLOC_PREF_META_NO`;
/// - DO NOT hard code any values; and
/// - DO NOT mix this/cast this with/to `u8`, `u16`, `usize` (nor any other integer).
#[unstable(feature = "global_co_alloc_meta", issue = "none")]
pub type CoAllocPref = usize; //u8;

/// `CoAllocMetaNumSlotsPref` values indicate that a type (but not necessarily an allocator) prefers
/// to coallocate by carrying metadata, or not. (In either user space, or `std` or `alloc` space).
/// Used as an argument to macro call of `co_alloc_pref`, which generates a `CoAllocPref` value.
///
/// Currently this indicates only the (preferred) number of `CoAllocMetaBase` slots being used
/// (either 1 = coallocation, or 0 = no coallocation). However, in the future this type may have
/// other properties (serving as extra hints to the allocator).
///
/// The actual value may be overriden by the allocator. For example, if the allocator doesn't
/// support coallocation, then whether this value prefers to coallocate or not makes no difference.
///
/// This type  WILL CHANGE (once ``#![feature(generic_const_exprs)]` and
/// `#![feature(adt_const_params)]` are stable) to a dedicated struct/enum. Hence:
/// - DO NOT mix this/cast this with/to `u8`, `u16`, (nor any other integer); and
/// - DO NOT hard code any values, but use `CO_ALLOC_PREF_META_YES` and `CO_ALLOC_PREF_META_NO`.
///
/// This type is intentionally not `u16`, `u32`, nor `usize`. Why? This helps to prevent mistakes
/// when one would use `CO_ALLOC_PREF_META_YES` or `CO_ALLOC_PREF_META_NO` in place of `CoAllocPref`
/// vales, or in place of a result of `meta_num_slots` macro. That also prevents mixing up with
/// [core::alloc::CoAllocatorMetaNumSlots].
#[unstable(feature = "global_co_alloc_meta", issue = "none")]
pub type CoAllocMetaNumSlotsPref = u16;
