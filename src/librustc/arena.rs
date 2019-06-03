use arena::{TypedArena, DroplessArena};
use std::mem;
use std::ptr;
use std::slice;
use std::cell::RefCell;
use std::marker::PhantomData;
use smallvec::SmallVec;

/// This declares a list of types which can be allocated by `Arena`.
///
/// The `few` modifier will cause allocation to use the shared arena and recording the destructor.
/// This is faster and more memory efficient if there's only a few allocations of the type.
/// Leaving `few` out will cause the type to get its own dedicated `TypedArena` which is
/// faster and more memory efficient if there is lots of allocations.
///
/// Specifying the `decode` modifier will add decode impls for &T and &[T] where T is the type
/// listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            [] layouts: rustc::ty::layout::LayoutDetails,
            [] generics: rustc::ty::Generics,
            [] trait_def: rustc::ty::TraitDef,
            [] adt_def: rustc::ty::AdtDef,
            [] steal_mir: rustc::ty::steal::Steal<rustc::mir::Body<$tcx>>,
            [] mir: rustc::mir::Body<$tcx>,
            [] tables: rustc::ty::TypeckTables<$tcx>,
            [] const_allocs: rustc::mir::interpret::Allocation,
            [] vtable_method: Option<(
                rustc::hir::def_id::DefId,
                rustc::ty::subst::SubstsRef<$tcx>
            )>,
            [few, decode] mir_keys: rustc::util::nodemap::DefIdSet,
            [decode] specialization_graph: rustc::traits::specialization_graph::Graph,
            [] region_scope_tree: rustc::middle::region::ScopeTree,
            [] item_local_set: rustc::util::nodemap::ItemLocalSet,
            [decode] mir_const_qualif: rustc_data_structures::bit_set::BitSet<rustc::mir::Local>,
            [] trait_impls_of: rustc::ty::trait_def::TraitImpls,
            [] dropck_outlives:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        rustc::traits::query::dropck_outlives::DropckOutlivesResult<'tcx>
                    >
                >,
            [] normalize_projection_ty:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        rustc::traits::query::normalize::NormalizationResult<'tcx>
                    >
                >,
            [] implied_outlives_bounds:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        Vec<rustc::traits::query::outlives_bounds::OutlivesBound<'tcx>>
                    >
                >,
            [] type_op_subtype:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, ()>
                >,
            [] type_op_normalize_poly_fn_sig:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::PolyFnSig<'tcx>>
                >,
            [] type_op_normalize_fn_sig:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::FnSig<'tcx>>
                >,
            [] type_op_normalize_predicate:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::Predicate<'tcx>>
                >,
            [] type_op_normalize_ty:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::Ty<'tcx>>
                >,
            [few] crate_inherent_impls: rustc::ty::CrateInherentImpls,
            [decode] borrowck: rustc::middle::borrowck::BorrowCheckResult,
            [few] upstream_monomorphizations:
                rustc::util::nodemap::DefIdMap<
                    rustc_data_structures::fx::FxHashMap<
                        rustc::ty::subst::SubstsRef<'tcx>,
                        rustc::hir::def_id::CrateNum
                    >
                >,
            [few] resolve_lifetimes: rustc::middle::resolve_lifetime::ResolveLifetimes,
            [decode] generic_predicates: rustc::ty::GenericPredicates<'tcx>,
            [few] lint_levels: rustc::lint::LintLevelMap,
            [few] stability_index: rustc::middle::stability::Index<'tcx>,
            [few] features: syntax::feature_gate::Features,
            [few] all_traits: Vec<rustc::hir::def_id::DefId>,
            [few] privacy_access_levels: rustc::middle::privacy::AccessLevels,
            [few] target_features_whitelist: rustc_data_structures::fx::FxHashMap<
                String,
                Option<syntax::symbol::Symbol>
            >,
            [few] wasm_import_module_map: rustc_data_structures::fx::FxHashMap<
                rustc::hir::def_id::DefId,
                String
            >,
            [few] get_lib_features: rustc::middle::lib_features::LibFeatures,
            [few] defined_lib_features: rustc::middle::lang_items::LanguageItems,
            [few] visible_parent_map: rustc::util::nodemap::DefIdMap<rustc::hir::def_id::DefId>,
            [few] foreign_module: rustc::middle::cstore::ForeignModule,
            [few] foreign_modules: Vec<rustc::middle::cstore::ForeignModule>,
            [few] reachable_non_generics: rustc::util::nodemap::DefIdMap<
                rustc::middle::exported_symbols::SymbolExportLevel
            >,
            [few] crate_variances: rustc::ty::CrateVariancesMap<'tcx>,
            [few] inferred_outlives_crate: rustc::ty::CratePredicatesMap<'tcx>,
            [] upvars: rustc_data_structures::fx::FxIndexMap<rustc::hir::HirId, rustc::hir::Upvar>,
        ], $tcx);
    )
}

macro_rules! arena_for_type {
    ([][$ty:ty]) => {
        TypedArena<$ty>
    };
    ([few $(, $attrs:ident)*][$ty:ty]) => {
        PhantomData<$ty>
    };
    ([$ignore:ident $(, $attrs:ident)*]$args:tt) => {
        arena_for_type!([$($attrs),*]$args)
    };
}

macro_rules! declare_arena {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        #[derive(Default)]
        pub struct Arena<$tcx> {
            dropless: DroplessArena,
            drop: DropArena,
            $($name: arena_for_type!($a[$ty]),)*
        }
    }
}

macro_rules! which_arena_for_type {
    ([][$arena:expr]) => {
        Some($arena)
    };
    ([few$(, $attrs:ident)*][$arena:expr]) => {
        None
    };
    ([$ignore:ident$(, $attrs:ident)*]$args:tt) => {
        which_arena_for_type!([$($attrs),*]$args)
    };
}

macro_rules! impl_arena_allocatable {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl ArenaAllocatable for $ty {}
            unsafe impl<$tcx> ArenaField<$tcx> for $ty {
                #[inline]
                fn arena<'a>(_arena: &'a Arena<$tcx>) -> Option<&'a TypedArena<Self>> {
                    which_arena_for_type!($a[&_arena.$name])
                }
            }
        )*
    }
}

arena_types!(declare_arena, [], 'tcx);

arena_types!(impl_arena_allocatable, [], 'tcx);

pub trait ArenaAllocatable {}

impl<T: Copy> ArenaAllocatable for T {}

unsafe trait ArenaField<'tcx>: Sized {
    /// Returns a specific arena to allocate from.
    /// If None is returned, the DropArena will be used.
    fn arena<'a>(arena: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>>;
}

unsafe impl<'tcx, T> ArenaField<'tcx> for T {
    #[inline]
    default fn arena<'a>(_: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>> {
        panic!()
    }
}

impl<'tcx> Arena<'tcx> {
    #[inline]
    pub fn alloc<T: ArenaAllocatable>(&self, value: T) -> &mut T {
        if !mem::needs_drop::<T>() {
            return self.dropless.alloc(value);
        }
        match <T as ArenaField<'tcx>>::arena(self) {
            Some(arena) => arena.alloc(value),
            None => unsafe { self.drop.alloc(value) },
        }
    }

    #[inline]
    pub fn alloc_slice<T: Copy>(&self, value: &[T]) -> &mut [T] {
        if value.len() == 0 {
            return &mut []
        }
        self.dropless.alloc_slice(value)
    }

    pub fn alloc_from_iter<
        T: ArenaAllocatable,
        I: IntoIterator<Item = T>
    >(
        &'a self,
        iter: I
    ) -> &'a mut [T] {
        if !mem::needs_drop::<T>() {
            return self.dropless.alloc_from_iter(iter);
        }
        match <T as ArenaField<'tcx>>::arena(self) {
            Some(arena) => arena.alloc_from_iter(iter),
            None => unsafe { self.drop.alloc_from_iter(iter) },
        }
    }
}

/// Calls the destructor for an object when dropped.
struct DropType {
    drop_fn: unsafe fn(*mut u8),
    obj: *mut u8,
}

unsafe fn drop_for_type<T>(to_drop: *mut u8) {
    std::ptr::drop_in_place(to_drop as *mut T)
}

impl Drop for DropType {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.obj)
        }
    }
}

/// An arena which can be used to allocate any type.
/// Allocating in this arena is unsafe since the type system
/// doesn't know which types it contains. In order to
/// allocate safely, you must store a PhantomData<T>
/// alongside this arena for each type T you allocate.
#[derive(Default)]
struct DropArena {
    /// A list of destructors to run when the arena drops.
    /// Ordered so `destructors` gets dropped before the arena
    /// since its destructor can reference memory in the arena.
    destructors: RefCell<Vec<DropType>>,
    arena: DroplessArena,
}

impl DropArena {
    #[inline]
    unsafe fn alloc<T>(&self, object: T) -> &mut T {
        let mem = self.arena.alloc_raw(
            mem::size_of::<T>(),
            mem::align_of::<T>()
        ) as *mut _ as *mut T;
        // Write into uninitialized memory.
        ptr::write(mem, object);
        let result = &mut *mem;
        // Record the destructor after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        self.destructors.borrow_mut().push(DropType {
            drop_fn: drop_for_type::<T>,
            obj: result as *mut T as *mut u8,
        });
        result
    }

    #[inline]
    unsafe fn alloc_from_iter<T, I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        let mut vec: SmallVec<[_; 8]> = iter.into_iter().collect();
        if vec.is_empty() {
            return &mut [];
        }
        let len = vec.len();

        let start_ptr = self.arena.alloc_raw(
            len.checked_mul(mem::size_of::<T>()).unwrap(),
            mem::align_of::<T>()
        ) as *mut _ as *mut T;

        let mut destructors = self.destructors.borrow_mut();
        // Reserve space for the destructors so we can't panic while adding them
        destructors.reserve(len);

        // Move the content to the arena by copying it and then forgetting
        // the content of the SmallVec
        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
        mem::forget(vec.drain());

        // Record the destructors after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        for i in 0..len {
            destructors.push(DropType {
                drop_fn: drop_for_type::<T>,
                obj: start_ptr.offset(i as isize) as *mut u8,
            });
        }

        slice::from_raw_parts_mut(start_ptr, len)
    }
}
