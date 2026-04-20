//! Global `Arc`-based object interning infrastructure.
//!
//! Eventually this should probably be replaced with salsa-based interning.

mod gc;
mod intern;
mod intern_slice;
mod symbol;

extern crate ra_ap_rustc_type_ir as rustc_type_ir;

pub use self::gc::{GarbageCollector, GcInternedSliceVisit, GcInternedVisit};
pub use self::intern::{InternStorage, Internable, Interned, InternedRef, impl_internable};
pub use self::intern_slice::{
    InternSliceStorage, InternedSlice, InternedSliceRef, SliceInternable, impl_slice_internable,
};
pub use self::symbol::{Symbol, symbols as sym};
