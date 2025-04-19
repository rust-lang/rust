//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.
//!
//! In principle, this is not very complicated: we recursively walk the final value, follow all the
//! pointers, and move all reachable allocations to the global `tcx` memory. The only complication
//! is picking the right mutability: the outermost allocation generally has a clear mutability, but
//! what about the other allocations it points to that have also been created with this value? We
//! don't want to do guesswork here. The rules are: `static`, `const`, and promoted can only create
//! immutable allocations that way. `static mut` can be initialized with expressions like `&mut 42`,
//! so all inner allocations are marked mutable. Some of them could potentially be made immutable,
//! but that would require relying on type information, and given how many ways Rust has to lie
//! about type information, we want to avoid doing that.

use hir::def::DefKind;
use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir as hir;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::interpret::{ConstAllocation, CtfeProvenance, InterpResult};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_span::def_id::LocalDefId;
use rustc_span::sym;
use tracing::{instrument, trace};

use super::{
    AllocId, Allocation, InterpCx, MPlaceTy, Machine, MemoryKind, PlaceTy, err_ub, interp_ok,
};
use crate::const_eval;
use crate::const_eval::DummyMachine;
use crate::errors::NestedStaticInThreadLocal;

pub trait CompileTimeMachine<'tcx, T> = Machine<
        'tcx,
        MemoryKind = T,
        Provenance = CtfeProvenance,
        ExtraFnVal = !,
        FrameExtra = (),
        AllocExtra = (),
        MemoryMap = FxIndexMap<AllocId, (MemoryKind<T>, Allocation)>,
    > + HasStaticRootDefId;

pub trait HasStaticRootDefId {
    /// Returns the `DefId` of the static item that is currently being evaluated.
    /// Used for interning to be able to handle nested allocations.
    fn static_def_id(&self) -> Option<LocalDefId>;
}

impl HasStaticRootDefId for const_eval::CompileTimeMachine<'_> {
    fn static_def_id(&self) -> Option<LocalDefId> {
        Some(self.static_root_ids?.1)
    }
}

/// Intern an allocation. Returns `Err` if the allocation does not exist in the local memory.
///
/// `mutability` can be used to force immutable interning: if it is `Mutability::Not`, the
/// allocation is interned immutably; if it is `Mutability::Mut`, then the allocation *must be*
/// already mutable (as a sanity check).
///
/// Returns an iterator over all relocations referred to by this allocation.
fn intern_shallow<'tcx, T, M: CompileTimeMachine<'tcx, T>>(
    ecx: &mut InterpCx<'tcx, M>,
    alloc_id: AllocId,
    mutability: Mutability,
) -> Result<impl Iterator<Item = CtfeProvenance> + 'tcx, ()> {
    trace!("intern_shallow {:?}", alloc_id);
    // remove allocation
    // FIXME(#120456) - is `swap_remove` correct?
    let Some((_kind, mut alloc)) = ecx.memory.alloc_map.swap_remove(&alloc_id) else {
        return Err(());
    };
    // Set allocation mutability as appropriate. This is used by LLVM to put things into
    // read-only memory, and also by Miri when evaluating other globals that
    // access this one.
    match mutability {
        Mutability::Not => {
            alloc.mutability = Mutability::Not;
        }
        Mutability::Mut => {
            // This must be already mutable, we won't "un-freeze" allocations ever.
            assert_eq!(alloc.mutability, Mutability::Mut);
        }
    }
    // link the alloc id to the actual allocation
    let alloc = ecx.tcx.mk_const_alloc(alloc);
    if let Some(static_id) = ecx.machine.static_def_id() {
        intern_as_new_static(ecx.tcx, static_id, alloc_id, alloc);
    } else {
        ecx.tcx.set_alloc_id_memory(alloc_id, alloc);
    }
    Ok(alloc.0.0.provenance().ptrs().iter().map(|&(_, prov)| prov))
}

/// Creates a new `DefId` and feeds all the right queries to make this `DefId`
/// appear as if it were a user-written `static` (though it has no HIR).
fn intern_as_new_static<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    static_id: LocalDefId,
    alloc_id: AllocId,
    alloc: ConstAllocation<'tcx>,
) {
    let feed = tcx.create_def(
        static_id,
        Some(sym::nested),
        DefKind::Static { safety: hir::Safety::Safe, mutability: alloc.0.mutability, nested: true },
    );
    tcx.set_nested_alloc_id_static(alloc_id, feed.def_id());

    if tcx.is_thread_local_static(static_id.into()) {
        tcx.dcx().emit_err(NestedStaticInThreadLocal { span: tcx.def_span(static_id) });
    }

    // These do not inherit the codegen attrs of the parent static allocation, since
    // it doesn't make sense for them to inherit their `#[no_mangle]` and `#[link_name = ..]`
    // and the like.
    feed.codegen_fn_attrs(CodegenFnAttrs::new());

    feed.eval_static_initializer(Ok(alloc));
    feed.generics_of(tcx.generics_of(static_id).clone());
    feed.def_ident_span(tcx.def_ident_span(static_id));
    feed.explicit_predicates_of(tcx.explicit_predicates_of(static_id));
    feed.feed_hir();
}

/// How a constant value should be interned.
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
pub enum InternKind {
    /// The `mutability` of the static, ignoring the type which may have interior mutability.
    Static(hir::Mutability),
    /// A `const` item
    Constant,
    Promoted,
}

#[derive(Debug)]
pub enum InternResult {
    FoundBadMutablePointer,
    FoundDanglingPointer,
}

/// Intern `ret` and everything it references.
///
/// This *cannot raise an interpreter error*. Doing so is left to validation, which
/// tracks where in the value we are and thus can show much better error messages.
///
/// For `InternKind::Static` the root allocation will not be interned, but must be handled by the caller.
#[instrument(level = "debug", skip(ecx))]
pub fn intern_const_alloc_recursive<'tcx, M: CompileTimeMachine<'tcx, const_eval::MemoryKind>>(
    ecx: &mut InterpCx<'tcx, M>,
    intern_kind: InternKind,
    ret: &MPlaceTy<'tcx>,
) -> Result<(), InternResult> {
    // We are interning recursively, and for mutability we are distinguishing the "root" allocation
    // that we are starting in, and all other allocations that we are encountering recursively.
    let (base_mutability, inner_mutability, is_static) = match intern_kind {
        InternKind::Constant | InternKind::Promoted => {
            // Completely immutable. Interning anything mutably here can only lead to unsoundness,
            // since all consts are conceptually independent values but share the same underlying
            // memory.
            (Mutability::Not, Mutability::Not, false)
        }
        InternKind::Static(Mutability::Not) => {
            (
                // Outermost allocation is mutable if `!Freeze`.
                if ret.layout.ty.is_freeze(*ecx.tcx, ecx.typing_env) {
                    Mutability::Not
                } else {
                    Mutability::Mut
                },
                // Inner allocations are never mutable. They can only arise via the "tail
                // expression" / "outer scope" rule, and we treat them consistently with `const`.
                Mutability::Not,
                true,
            )
        }
        InternKind::Static(Mutability::Mut) => {
            // Just make everything mutable. We accept code like
            // `static mut X = &mut [42]`, so even inner allocations need to be mutable.
            (Mutability::Mut, Mutability::Mut, true)
        }
    };

    // Intern the base allocation, and initialize todo list for recursive interning.
    let base_alloc_id = ret.ptr().provenance.unwrap().alloc_id();
    trace!(?base_alloc_id, ?base_mutability);
    // First we intern the base allocation, as it requires a different mutability.
    // This gives us the initial set of nested allocations, which will then all be processed
    // recursively in the loop below.
    let mut todo: Vec<_> = if is_static {
        // Do not steal the root allocation, we need it later to create the return value of `eval_static_initializer`.
        // But still change its mutability to match the requested one.
        let alloc = ecx.memory.alloc_map.get_mut(&base_alloc_id).unwrap();
        alloc.1.mutability = base_mutability;
        alloc.1.provenance().ptrs().iter().map(|&(_, prov)| prov).collect()
    } else {
        intern_shallow(ecx, base_alloc_id, base_mutability).unwrap().collect()
    };
    // We need to distinguish "has just been interned" from "was already in `tcx`",
    // so we track this in a separate set.
    let mut just_interned: FxHashSet<_> = std::iter::once(base_alloc_id).collect();
    // Whether we encountered a bad mutable pointer.
    // We want to first report "dangling" and then "mutable", so we need to delay reporting these
    // errors.
    let mut result = Ok(());

    // Keep interning as long as there are things to intern.
    // We show errors if there are dangling pointers, or mutable pointers in immutable contexts
    // (i.e., everything except for `static mut`). When these errors affect references, it is
    // unfortunate that we show these errors here and not during validation, since validation can
    // show much nicer errors. However, we do need these checks to be run on all pointers, including
    // raw pointers, so we cannot rely on validation to catch them -- and since interning runs
    // before validation, and interning doesn't know the type of anything, this means we can't show
    // better errors. Maybe we should consider doing validation before interning in the future.
    while let Some(prov) = todo.pop() {
        trace!(?prov);
        let alloc_id = prov.alloc_id();

        if base_alloc_id == alloc_id && is_static {
            // This is a pointer to the static itself. It's ok for a static to refer to itself,
            // even mutably. Whether that mutable pointer is legal at all is checked in validation.
            // See tests/ui/statics/recursive_interior_mut.rs for how such a situation can occur.
            // We also already collected all the nested allocations, so there's no need to do that again.
            continue;
        }

        // Ensure that this is derived from a shared reference. Crucially, we check this *before*
        // checking whether the `alloc_id` has already been interned. The point of this check is to
        // ensure that when there are multiple pointers to the same allocation, they are *all*
        // derived from a shared reference. Therefore it would be bad if we only checked the first
        // pointer to any given allocation.
        // (It is likely not possible to actually have multiple pointers to the same allocation,
        // so alternatively we could also check that and ICE if there are multiple such pointers.)
        // See <https://github.com/rust-lang/rust/pull/128543> for why we are checking for "shared
        // reference" and not "immutable", i.e., for why we are allowing interior-mutable shared
        // references: they can actually be created in safe code while pointing to apparently
        // "immutable" values, via promotion or tail expression lifetime extension of
        // `&None::<Cell<T>>`.
        // We also exclude promoteds from this as `&mut []` can be promoted, which is a mutable
        // reference pointing to an immutable (zero-sized) allocation. We rely on the promotion
        // analysis not screwing up to ensure that it is sound to intern promoteds as immutable.
        if intern_kind != InternKind::Promoted
            && inner_mutability == Mutability::Not
            && !prov.shared_ref()
        {
            let is_already_global = ecx.tcx.try_get_global_alloc(alloc_id).is_some();
            if is_already_global && !just_interned.contains(&alloc_id) {
                // This is a pointer to some memory from another constant. We encounter mutable
                // pointers to such memory since we do not always track immutability through
                // these "global" pointers. Allowing them is harmless; the point of these checks
                // during interning is to justify why we intern the *new* allocations immutably,
                // so we can completely ignore existing allocations.
                // We can also skip the rest of this loop iteration, since after all it is already
                // interned.
                continue;
            }
            // If this is a dangling pointer, that's actually fine -- the problematic case is
            // when there is memory there that someone might expect to be mutable, but we make it immutable.
            let dangling = !is_already_global && !ecx.memory.alloc_map.contains_key(&alloc_id);
            if !dangling {
                // Found a mutable reference inside a const where inner allocations should be
                // immutable.
                if !ecx.tcx.sess.opts.unstable_opts.unleash_the_miri_inside_of_you {
                    span_bug!(
                        ecx.tcx.span,
                        "the static const safety checks accepted mutable references they should not have accepted"
                    );
                }
                // Prefer dangling pointer errors over mutable pointer errors
                if result.is_ok() {
                    result = Err(InternResult::FoundBadMutablePointer);
                }
            }
        }
        if ecx.tcx.try_get_global_alloc(alloc_id).is_some() {
            // Already interned.
            debug_assert!(!ecx.memory.alloc_map.contains_key(&alloc_id));
            continue;
        }
        // We always intern with `inner_mutability`, and furthermore we ensured above that if
        // that is "immutable", then there are *no* mutable pointers anywhere in the newly
        // interned memory -- justifying that we can indeed intern immutably. However this also
        // means we can *not* easily intern immutably here if `prov.immutable()` is true and
        // `inner_mutability` is `Mut`: there might be other pointers to that allocation, and
        // we'd have to somehow check that they are *all* immutable before deciding that this
        // allocation can be made immutable. In the future we could consider analyzing all
        // pointers before deciding which allocations can be made immutable; but for now we are
        // okay with losing some potential for immutability here. This can anyway only affect
        // `static mut`.
        just_interned.insert(alloc_id);
        match intern_shallow(ecx, alloc_id, inner_mutability) {
            Ok(nested) => todo.extend(nested),
            Err(()) => {
                ecx.tcx.dcx().delayed_bug("found dangling pointer during const interning");
                result = Err(InternResult::FoundDanglingPointer);
            }
        }
    }
    result
}

/// Intern `ret`. This function assumes that `ret` references no other allocation.
#[instrument(level = "debug", skip(ecx))]
pub fn intern_const_alloc_for_constprop<'tcx, T, M: CompileTimeMachine<'tcx, T>>(
    ecx: &mut InterpCx<'tcx, M>,
    alloc_id: AllocId,
) -> InterpResult<'tcx, ()> {
    if ecx.tcx.try_get_global_alloc(alloc_id).is_some() {
        // The constant is already in global memory. Do nothing.
        return interp_ok(());
    }
    // Move allocation to `tcx`.
    if let Some(_) =
        (intern_shallow(ecx, alloc_id, Mutability::Not).map_err(|()| err_ub!(DeadLocal))?).next()
    {
        // We are not doing recursive interning, so we don't currently support provenance.
        // (If this assertion ever triggers, we should just implement a
        // proper recursive interning loop -- or just call `intern_const_alloc_recursive`.
        panic!("`intern_const_alloc_for_constprop` called on allocation with nested provenance")
    }
    interp_ok(())
}

impl<'tcx> InterpCx<'tcx, DummyMachine> {
    /// A helper function that allocates memory for the layout given and gives you access to mutate
    /// it. Once your own mutation code is done, the backing `Allocation` is removed from the
    /// current `Memory` and interned as read-only into the global memory.
    pub fn intern_with_temp_alloc(
        &mut self,
        layout: TyAndLayout<'tcx>,
        f: impl FnOnce(
            &mut InterpCx<'tcx, DummyMachine>,
            &PlaceTy<'tcx, CtfeProvenance>,
        ) -> InterpResult<'tcx, ()>,
    ) -> InterpResult<'tcx, AllocId> {
        // `allocate` picks a fresh AllocId that we will associate with its data below.
        let dest = self.allocate(layout, MemoryKind::Stack)?;
        f(self, &dest.clone().into())?;
        let alloc_id = dest.ptr().provenance.unwrap().alloc_id(); // this was just allocated, it must have provenance
        for prov in intern_shallow(self, alloc_id, Mutability::Not).unwrap() {
            // We are not doing recursive interning, so we don't currently support provenance.
            // (If this assertion ever triggers, we should just implement a
            // proper recursive interning loop -- or just call `intern_const_alloc_recursive`.
            if self.tcx.try_get_global_alloc(prov.alloc_id()).is_none() {
                panic!("`intern_with_temp_alloc` with nested allocations");
            }
        }
        interp_ok(alloc_id)
    }
}
