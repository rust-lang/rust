//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.
//!
//! In principle, this is not very complicated: we recursively walk the final value, follow all the
//! pointers, and move all reachable allocations to the global `tcx` memory. The only complication
//! is picking the right mutability for the allocations in a `static` initializer: we want to make
//! as many allocations as possible immutable so LLVM can put them into read-only memory. At the
//! same time, we need to make memory that could be mutated by the program mutable to avoid
//! incorrect compilations. To achieve this, we do a type-based traversal of the final value,
//! tracking mutable and shared references and `UnsafeCell` to determine the current mutability.
//! (In principle, we could skip this type-based part for `const` and promoteds, as they need to be
//! always immutable. At least for `const` however we use this opportunity to reject any `const`
//! that contains allocations whose mutability we cannot identify.)

use super::validity::RefTracking;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty::{self, layout::TyAndLayout, Ty};

use rustc_ast::Mutability;

use super::{
    AllocId, Allocation, ConstAllocation, InterpCx, MPlaceTy, Machine, MemoryKind, PlaceTy,
    ValueVisitor,
};
use crate::const_eval;

pub trait CompileTimeMachine<'mir, 'tcx, T> = Machine<
    'mir,
    'tcx,
    MemoryKind = T,
    Provenance = AllocId,
    ExtraFnVal = !,
    FrameExtra = (),
    AllocExtra = (),
    MemoryMap = FxHashMap<AllocId, (MemoryKind<T>, Allocation)>,
>;

struct InternVisitor<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx, const_eval::MemoryKind>> {
    /// The ectx from which we intern.
    ecx: &'rt mut InterpCx<'mir, 'tcx, M>,
    /// Previously encountered safe references.
    ref_tracking: &'rt mut RefTracking<(MPlaceTy<'tcx>, InternMode)>,
    /// A list of all encountered allocations. After type-based interning, we traverse this list to
    /// also intern allocations that are only referenced by a raw pointer or inside a union.
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    /// The root kind of the value that we're looking at. This field is never mutated for a
    /// particular allocation. It is primarily used to make as many allocations as possible
    /// read-only so LLVM can place them in const memory.
    mode: InternMode,
    /// This field stores whether we are *currently* inside an `UnsafeCell`. This can affect
    /// the intern mode of references we encounter.
    inside_unsafe_cell: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
enum InternMode {
    /// A static and its current mutability.  Below shared references inside a `static mut`,
    /// this is *immutable*, and below mutable references inside an `UnsafeCell`, this
    /// is *mutable*.
    Static(hir::Mutability),
    /// A `const`.
    Const,
}

/// Signalling data structure to ensure we don't recurse
/// into the memory of other constants or statics
struct IsStaticOrFn;

/// Intern an allocation without looking at its children.
/// `mode` is the mode of the environment where we found this pointer.
/// `mutability` is the mutability of the place to be interned; even if that says
/// `immutable` things might become mutable if `ty` is not frozen.
/// `ty` can be `None` if there is no potential interior mutability
/// to account for (e.g. for vtables).
fn intern_shallow<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx, const_eval::MemoryKind>>(
    ecx: &'rt mut InterpCx<'mir, 'tcx, M>,
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    alloc_id: AllocId,
    mode: InternMode,
    ty: Option<Ty<'tcx>>,
) -> Option<IsStaticOrFn> {
    trace!("intern_shallow {:?} with {:?}", alloc_id, mode);
    // remove allocation
    let tcx = ecx.tcx;
    let Some((kind, mut alloc)) = ecx.memory.alloc_map.remove(&alloc_id) else {
        // Pointer not found in local memory map. It is either a pointer to the global
        // map, or dangling.
        // If the pointer is dangling (neither in local nor global memory), we leave it
        // to validation to error -- it has the much better error messages, pointing out where
        // in the value the dangling reference lies.
        // The `delay_span_bug` ensures that we don't forget such a check in validation.
        if tcx.try_get_global_alloc(alloc_id).is_none() {
            tcx.sess.delay_span_bug(ecx.tcx.span, "tried to intern dangling pointer");
        }
        // treat dangling pointers like other statics
        // just to stop trying to recurse into them
        return Some(IsStaticOrFn);
    };
    // This match is just a canary for future changes to `MemoryKind`, which most likely need
    // changes in this function.
    match kind {
        MemoryKind::Stack
        | MemoryKind::Machine(const_eval::MemoryKind::Heap)
        | MemoryKind::CallerLocation => {}
    }
    // Set allocation mutability as appropriate. This is used by LLVM to put things into
    // read-only memory, and also by Miri when evaluating other globals that
    // access this one.
    if let InternMode::Static(mutability) = mode {
        // For this, we need to take into account `UnsafeCell`. When `ty` is `None`, we assume
        // no interior mutability.
        let frozen = ty.map_or(true, |ty| ty.is_freeze(ecx.tcx, ecx.param_env));
        // For statics, allocation mutability is the combination of place mutability and
        // type mutability.
        // The entire allocation needs to be mutable if it contains an `UnsafeCell` anywhere.
        let immutable = mutability == Mutability::Not && frozen;
        if immutable {
            alloc.mutability = Mutability::Not;
        } else {
            // Just making sure we are not "upgrading" an immutable allocation to mutable.
            assert_eq!(alloc.mutability, Mutability::Mut);
        }
    } else {
        // No matter what, *constants are never mutable*. Mutating them is UB.
        // See const_eval::machine::MemoryExtra::can_access_statics for why
        // immutability is so important.

        // Validation will ensure that there is no `UnsafeCell` on an immutable allocation.
        alloc.mutability = Mutability::Not;
    };
    // link the alloc id to the actual allocation
    leftover_allocations.extend(alloc.provenance().iter().map(|&(_, alloc_id)| alloc_id));
    let alloc = tcx.intern_const_alloc(alloc);
    tcx.set_alloc_id_memory(alloc_id, alloc);
    None
}

impl<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx, const_eval::MemoryKind>>
    InternVisitor<'rt, 'mir, 'tcx, M>
{
    fn intern_shallow(
        &mut self,
        alloc_id: AllocId,
        mode: InternMode,
        ty: Option<Ty<'tcx>>,
    ) -> Option<IsStaticOrFn> {
        intern_shallow(self.ecx, self.leftover_allocations, alloc_id, mode, ty)
    }
}

impl<'rt, 'mir, 'tcx: 'mir, M: CompileTimeMachine<'mir, 'tcx, const_eval::MemoryKind>>
    ValueVisitor<'mir, 'tcx, M> for InternVisitor<'rt, 'mir, 'tcx, M>
{
    type V = MPlaceTy<'tcx>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'mir, 'tcx, M> {
        &self.ecx
    }

    fn visit_aggregate(
        &mut self,
        mplace: &MPlaceTy<'tcx>,
        fields: impl Iterator<Item = InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        // We want to walk the aggregate to look for references to intern. While doing that we
        // also need to take special care of interior mutability.
        //
        // As an optimization, however, if the allocation does not contain any references: we don't
        // need to do the walk. It can be costly for big arrays for example (e.g. issue #93215).
        let is_walk_needed = |mplace: &MPlaceTy<'tcx>| -> InterpResult<'tcx, bool> {
            // ZSTs cannot contain pointers, we can avoid the interning walk.
            if mplace.layout.is_zst() {
                return Ok(false);
            }

            // Now, check whether this allocation could contain references.
            //
            // Note, this check may sometimes not be cheap, so we only do it when the walk we'd like
            // to avoid could be expensive: on the potentially larger types, arrays and slices,
            // rather than on all aggregates unconditionally.
            if matches!(mplace.layout.ty.kind(), ty::Array(..) | ty::Slice(..)) {
                let Some((size, align)) = self.ecx.size_and_align_of_mplace(&mplace)? else {
                    // We do the walk if we can't determine the size of the mplace: we may be
                    // dealing with extern types here in the future.
                    return Ok(true);
                };

                // If there is no provenance in this allocation, it does not contain references
                // that point to another allocation, and we can avoid the interning walk.
                if let Some(alloc) = self.ecx.get_ptr_alloc(mplace.ptr, size, align)? {
                    if !alloc.has_provenance() {
                        return Ok(false);
                    }
                } else {
                    // We're encountering a ZST here, and can avoid the walk as well.
                    return Ok(false);
                }
            }

            // In the general case, we do the walk.
            Ok(true)
        };

        // If this allocation contains no references to intern, we avoid the potentially costly
        // walk.
        //
        // We can do this before the checks for interior mutability below, because only references
        // are relevant in that situation, and we're checking if there are any here.
        if !is_walk_needed(mplace)? {
            return Ok(());
        }

        if let Some(def) = mplace.layout.ty.ty_adt_def() {
            if def.is_unsafe_cell() {
                // We are crossing over an `UnsafeCell`, we can mutate again. This means that
                // References we encounter inside here are interned as pointing to mutable
                // allocations.
                // Remember the `old` value to handle nested `UnsafeCell`.
                let old = std::mem::replace(&mut self.inside_unsafe_cell, true);
                let walked = self.walk_aggregate(mplace, fields);
                self.inside_unsafe_cell = old;
                return walked;
            }
        }

        self.walk_aggregate(mplace, fields)
    }

    fn visit_value(&mut self, mplace: &MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        // Handle Reference types, as these are the only types with provenance supported by const eval.
        // Raw pointers (and boxes) are handled by the `leftover_allocations` logic.
        let tcx = self.ecx.tcx;
        let ty = mplace.layout.ty;
        if let ty::Ref(_, referenced_ty, ref_mutability) = *ty.kind() {
            let value = self.ecx.read_immediate(&mplace.into())?;
            let mplace = self.ecx.ref_to_mplace(&value)?;
            assert_eq!(mplace.layout.ty, referenced_ty);
            // Handle trait object vtables.
            if let ty::Dynamic(..) =
                tcx.struct_tail_erasing_lifetimes(referenced_ty, self.ecx.param_env).kind()
            {
                let ptr = mplace.meta.unwrap_meta().to_pointer(&tcx)?;
                if let Some(alloc_id) = ptr.provenance {
                    // Explicitly choose const mode here, since vtables are immutable, even
                    // if the reference of the fat pointer is mutable.
                    self.intern_shallow(alloc_id, InternMode::Const, None);
                } else {
                    // Validation will error (with a better message) on an invalid vtable pointer.
                    // Let validation show the error message, but make sure it *does* error.
                    tcx.sess
                        .delay_span_bug(tcx.span, "vtables pointers cannot be integer pointers");
                }
            }
            // Check if we have encountered this pointer+layout combination before.
            // Only recurse for allocation-backed pointers.
            if let Some(alloc_id) = mplace.ptr.provenance {
                // Compute the mode with which we intern this. Our goal here is to make as many
                // statics as we can immutable so they can be placed in read-only memory by LLVM.
                let ref_mode = match self.mode {
                    InternMode::Static(mutbl) => {
                        // In statics, merge outer mutability with reference mutability and
                        // take into account whether we are in an `UnsafeCell`.

                        // The only way a mutable reference actually works as a mutable reference is
                        // by being in a `static mut` directly or behind another mutable reference.
                        // If there's an immutable reference or we are inside a `static`, then our
                        // mutable reference is equivalent to an immutable one. As an example:
                        // `&&mut Foo` is semantically equivalent to `&&Foo`
                        match ref_mutability {
                            _ if self.inside_unsafe_cell => {
                                // Inside an `UnsafeCell` is like inside a `static mut`, the "outer"
                                // mutability does not matter.
                                InternMode::Static(ref_mutability)
                            }
                            Mutability::Not => {
                                // A shared reference, things become immutable.
                                // We do *not* consider `freeze` here: `intern_shallow` considers
                                // `freeze` for the actual mutability of this allocation; the intern
                                // mode for references contained in this allocation is tracked more
                                // precisely when traversing the referenced data (by tracking
                                // `UnsafeCell`). This makes sure that `&(&i32, &Cell<i32>)` still
                                // has the left inner reference interned into a read-only
                                // allocation.
                                InternMode::Static(Mutability::Not)
                            }
                            Mutability::Mut => {
                                // Mutable reference.
                                InternMode::Static(mutbl)
                            }
                        }
                    }
                    InternMode::Const => {
                        // Ignore `UnsafeCell`, everything is immutable.  Validity does some sanity
                        // checking for mutable references that we encounter -- they must all be
                        // ZST.
                        InternMode::Const
                    }
                };
                match self.intern_shallow(alloc_id, ref_mode, Some(referenced_ty)) {
                    // No need to recurse, these are interned already and statics may have
                    // cycles, so we don't want to recurse there
                    Some(IsStaticOrFn) => {}
                    // intern everything referenced by this value. The mutability is taken from the
                    // reference. It is checked above that mutable references only happen in
                    // `static mut`
                    None => self.ref_tracking.track((mplace, ref_mode), || ()),
                }
            }
            Ok(())
        } else {
            // Not a reference -- proceed recursively.
            self.walk_value(mplace)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
pub enum InternKind {
    /// The `mutability` of the static, ignoring the type which may have interior mutability.
    Static(hir::Mutability),
    Constant,
    Promoted,
}

/// Intern `ret` and everything it references.
///
/// This *cannot raise an interpreter error*.  Doing so is left to validation, which
/// tracks where in the value we are and thus can show much better error messages.
/// Any errors here would anyway be turned into `const_err` lints, whereas validation failures
/// are hard errors.
#[tracing::instrument(level = "debug", skip(ecx))]
pub fn intern_const_alloc_recursive<
    'mir,
    'tcx: 'mir,
    M: CompileTimeMachine<'mir, 'tcx, const_eval::MemoryKind>,
>(
    ecx: &mut InterpCx<'mir, 'tcx, M>,
    intern_kind: InternKind,
    ret: &MPlaceTy<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = ecx.tcx;
    let base_intern_mode = match intern_kind {
        InternKind::Static(mutbl) => InternMode::Static(mutbl),
        // `Constant` includes array lengths.
        InternKind::Constant | InternKind::Promoted => InternMode::Const,
    };

    // Type based interning.
    // `ref_tracking` tracks typed references we have already interned and still need to crawl for
    // more typed information inside them.
    // `leftover_allocations` collects *all* allocations we see, because some might not
    // be available in a typed way. They get interned at the end.
    let mut ref_tracking = RefTracking::empty();
    let leftover_allocations = &mut FxHashSet::default();

    // start with the outermost allocation
    intern_shallow(
        ecx,
        leftover_allocations,
        // The outermost allocation must exist, because we allocated it with
        // `Memory::allocate`.
        ret.ptr.provenance.unwrap(),
        base_intern_mode,
        Some(ret.layout.ty),
    );

    ref_tracking.track((*ret, base_intern_mode), || ());

    while let Some(((mplace, mode), _)) = ref_tracking.todo.pop() {
        let res = InternVisitor {
            ref_tracking: &mut ref_tracking,
            ecx,
            mode,
            leftover_allocations,
            inside_unsafe_cell: false,
        }
        .visit_value(&mplace);
        // We deliberately *ignore* interpreter errors here.  When there is a problem, the remaining
        // references are "leftover"-interned, and later validation will show a proper error
        // and point at the right part of the value causing the problem.
        match res {
            Ok(()) => {}
            Err(error) => {
                ecx.tcx.sess.delay_span_bug(
                    ecx.tcx.span,
                    &format!(
                        "error during interning should later cause validation failure: {}",
                        error
                    ),
                );
            }
        }
    }

    // Intern the rest of the allocations as mutable. These might be inside unions, padding, raw
    // pointers, ... So we can't intern them according to their type rules

    let mut todo: Vec<_> = leftover_allocations.iter().cloned().collect();
    debug!(?todo);
    debug!("dead_alloc_map: {:#?}", ecx.memory.dead_alloc_map);
    while let Some(alloc_id) = todo.pop() {
        if let Some((_, mut alloc)) = ecx.memory.alloc_map.remove(&alloc_id) {
            // We can't call the `intern_shallow` method here, as its logic is tailored to safe
            // references and a `leftover_allocations` set (where we only have a todo-list here).
            // So we hand-roll the interning logic here again.
            match intern_kind {
                // Statics may point to mutable allocations.
                // Even for immutable statics it would be ok to have mutable allocations behind
                // raw pointers, e.g. for `static FOO: *const AtomicUsize = &AtomicUsize::new(42)`.
                InternKind::Static(_) => {}
                // Raw pointers in promoteds may only point to immutable things so we mark
                // everything as immutable.
                // It is UB to mutate through a raw pointer obtained via an immutable reference:
                // Since all references and pointers inside a promoted must by their very definition
                // be created from an immutable reference (and promotion also excludes interior
                // mutability), mutating through them would be UB.
                // There's no way we can check whether the user is using raw pointers correctly,
                // so all we can do is mark this as immutable here.
                InternKind::Promoted => {
                    // See const_eval::machine::MemoryExtra::can_access_statics for why
                    // immutability is so important.
                    alloc.mutability = Mutability::Not;
                }
                InternKind::Constant => {
                    // If it's a constant, we should not have any "leftovers" as everything
                    // is tracked by const-checking.
                    // FIXME: downgrade this to a warning? It rejects some legitimate consts,
                    // such as `const CONST_RAW: *const Vec<i32> = &Vec::new() as *const _;`.
                    ecx.tcx
                        .sess
                        .span_err(ecx.tcx.span, "untyped pointers are not allowed in constant");
                    // For better errors later, mark the allocation as immutable.
                    alloc.mutability = Mutability::Not;
                }
            }
            let alloc = tcx.intern_const_alloc(alloc);
            tcx.set_alloc_id_memory(alloc_id, alloc);
            for &(_, alloc_id) in alloc.inner().provenance().iter() {
                if leftover_allocations.insert(alloc_id) {
                    todo.push(alloc_id);
                }
            }
        } else if ecx.memory.dead_alloc_map.contains_key(&alloc_id) {
            // Codegen does not like dangling pointers, and generally `tcx` assumes that
            // all allocations referenced anywhere actually exist. So, make sure we error here.
            let reported = ecx
                .tcx
                .sess
                .span_err(ecx.tcx.span, "encountered dangling pointer in final constant");
            return Err(reported);
        } else if ecx.tcx.try_get_global_alloc(alloc_id).is_none() {
            // We have hit an `AllocId` that is neither in local or global memory and isn't
            // marked as dangling by local memory.  That should be impossible.
            span_bug!(ecx.tcx.span, "encountered unknown alloc id {:?}", alloc_id);
        }
    }
    Ok(())
}

impl<'mir, 'tcx: 'mir, M: super::intern::CompileTimeMachine<'mir, 'tcx, !>>
    InterpCx<'mir, 'tcx, M>
{
    /// A helper function that allocates memory for the layout given and gives you access to mutate
    /// it. Once your own mutation code is done, the backing `Allocation` is removed from the
    /// current `Memory` and returned.
    pub fn intern_with_temp_alloc(
        &mut self,
        layout: TyAndLayout<'tcx>,
        f: impl FnOnce(
            &mut InterpCx<'mir, 'tcx, M>,
            &PlaceTy<'tcx, M::Provenance>,
        ) -> InterpResult<'tcx, ()>,
    ) -> InterpResult<'tcx, ConstAllocation<'tcx>> {
        let dest = self.allocate(layout, MemoryKind::Stack)?;
        f(self, &dest.into())?;
        let mut alloc = self.memory.alloc_map.remove(&dest.ptr.provenance.unwrap()).unwrap().1;
        alloc.mutability = Mutability::Not;
        Ok(self.tcx.intern_const_alloc(alloc))
    }
}
