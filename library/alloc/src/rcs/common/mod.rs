//! Provides infrastructures for reference counted types.
//!
//! A reference counted allocation contains the following parts in order:
//!
//! - Padding: allows header and value to be arranged consecutively while being properly aligned.
//!   We choose padding size of
//!   `size_of::<Header>().next_multiple_of(align_of::<Value>()) - size_of::<Header>()`.
//! - Header: stores reference counters.
//! - Value: the actual contained value.
//!
//! We the following layout for the allocation.
//!
//! - Alignment: `align_of::<Header>().max(align_of::<Value>())`.
//! - Size: `size_of::<Header>().next_multiple_of(align_of::<Value>()) + size_of::<Value>()`.
//!
//! Note that it is possible that the size of an allocation is not a multiple of its alignment.

#[cfg(not(no_global_oom_handling))]
pub(super) use rc_alloc::{
    allocate_from_box, allocate_from_cloning_in, allocate_from_iter, allocate_from_vec,
    allocate_uninit_in, allocate_with_in, allocate_zeroed_in,
};
pub(super) use rc_alloc::{
    deallocate, try_allocate_from_cloning_in, try_allocate_uninit_in, try_allocate_zeroed_in,
};
pub(super) use rc_layout::{RcLayout, RcLayoutExt};
pub(super) use rc_value_pointer::RcValuePointer;

mod rc_alloc;
mod rc_layout;
mod rc_value_pointer;
