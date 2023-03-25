//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(const_btree_len)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(variant_count)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

#[macro_use]
extern crate rustc_data_structures;

extern crate self as rustc_hir;

mod arena;
pub mod def;
pub mod def_path_hash_map;
pub mod definitions;
pub mod diagnostic_items;
pub mod errors;
use rustc_data_structures::thin_slice::ThinSlice;
pub use rustc_span::def_id;
mod hir;
pub mod hir_id;
pub mod intravisit;
pub mod lang_items;
pub mod pat_util;
mod stable_hash_impls;
mod target;
pub mod weak_lang_items;

#[cfg(test)]
mod tests;

pub use hir::*;
pub use hir_id::*;
pub use lang_items::{LangItem, LanguageItems};
use smallvec::SmallVec;
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};

arena_types!(rustc_arena::declare_arena);

pub trait ArenaAllocatableThin<'tcx, C = rustc_arena::IsNotCopy>: Sized {
    fn allocate_thin_from_iter<'a>(
        arena: &'a Arena<'tcx>,
        iter: impl ::std::iter::IntoIterator<Item = Self>,
    ) -> &'a ThinSlice<Self>;
}

// Any type that impls `Copy` can be arena-allocated in the `DroplessArena`.
impl<'tcx, T: Copy> ArenaAllocatableThin<'tcx, rustc_arena::IsCopy> for T {
    fn allocate_thin_from_iter<'a>(
        arena: &'a Arena<'tcx>,
        iter: impl ::std::iter::IntoIterator<Item = Self>,
    ) -> &'a ThinSlice<Self> {
        let mut vec: SmallVec<[_; 8]> = iter.into_iter().collect();

        let len = vec.len();
        if len == 0 {
            return ThinSlice::empty();
        }
        // Move the content to the arena by copying and then forgetting it.

        let slice = vec.as_slice();

        let (layout, _offset) = std::alloc::Layout::new::<usize>()
            .extend(std::alloc::Layout::for_value::<[T]>(slice))
            .unwrap();
        let mem = arena.dropless.alloc_raw(layout) as *mut ThinSlice<T>;
        // SAFETY: We ensured that we allocated enough memory above. It includes the ptr and the slice correctly aligned.
        let thin = unsafe { ThinSlice::initialize(mem, slice) };
        unsafe {
            vec.set_len(0);
        }
        thin
    }
}

impl<'tcx> Arena<'tcx> {
    #[inline]
    pub fn allocate_thin_from_iter<'a, T: ArenaAllocatableThin<'tcx, C>, C>(
        &'a self,
        iter: impl ::std::iter::IntoIterator<Item = T>,
    ) -> &'a ThinSlice<T> {
        T::allocate_thin_from_iter(self, iter)
    }
}
