//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.

use super::validity::RefTracking;
use rustc::mir::interpret::{ErrorHandled, InterpResult};
use rustc::ty::{self, Ty};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;

use syntax::ast::Mutability;

use super::{AllocId, Allocation, InterpCx, MPlaceTy, Machine, MemoryKind, Scalar, ValueVisitor};

pub trait CompileTimeMachine<'mir, 'tcx> = Machine<
    'mir,
    'tcx,
    MemoryKinds = !,
    PointerTag = (),
    ExtraFnVal = !,
    FrameExtra = (),
    AllocExtra = (),
    MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation)>,
>;

struct InternVisitor<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>> {
    /// The ectx from which we intern.
    ecx: &'rt mut InterpCx<'mir, 'tcx, M>,
    /// Previously encountered safe references.
    ref_tracking: &'rt mut RefTracking<(MPlaceTy<'tcx>, Mutability, InternMode)>,
    /// A list of all encountered allocations. After type-based interning, we traverse this list to
    /// also intern allocations that are only referenced by a raw pointer or inside a union.
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    /// The root node of the value that we're looking at. This field is never mutated and only used
    /// for sanity assertions that will ICE when `const_qualif` screws up.
    mode: InternMode,
    /// This field stores the mutability of the value *currently* being checked.
    /// When encountering a mutable reference, we determine the pointee mutability
    /// taking into account the mutability of the context: `& &mut i32` is entirely immutable,
    /// despite the nested mutable reference!
    /// The field gets updated when an `UnsafeCell` is encountered.
    mutability: Mutability,

    /// This flag is to avoid triggering UnsafeCells are not allowed behind references in constants
    /// for promoteds.
    /// It's a copy of `mir::Body`'s ignore_interior_mut_in_const_validation field
    ignore_interior_mut_in_const_validation: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
enum InternMode {
    /// Mutable references must in fact be immutable due to their surrounding immutability in a
    /// `static`. In a `static mut` we start out as mutable and thus can also contain further `&mut`
    /// that will actually be treated as mutable.
    Static,
    /// UnsafeCell is OK in the value of a constant: `const FOO = Cell::new(0)` creates
    /// a new cell every time it is used.
    ConstBase,
    /// `UnsafeCell` ICEs.
    Const,
}

/// Signalling data structure to ensure we don't recurse
/// into the memory of other constants or statics
struct IsStaticOrFn;

/// Intern an allocation without looking at its children.
/// `mode` is the mode of the environment where we found this pointer.
/// `mutablity` is the mutability of the place to be interned; even if that says
/// `immutable` things might become mutable if `ty` is not frozen.
/// `ty` can be `None` if there is no potential interior mutability
/// to account for (e.g. for vtables).
fn intern_shallow<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>>(
    ecx: &'rt mut InterpCx<'mir, 'tcx, M>,
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    mode: InternMode,
    alloc_id: AllocId,
    mutability: Mutability,
    ty: Option<Ty<'tcx>>,
) -> InterpResult<'tcx, Option<IsStaticOrFn>> {
    trace!("InternVisitor::intern {:?} with {:?}", alloc_id, mutability,);
    // remove allocation
    let tcx = ecx.tcx;
    let (kind, mut alloc) = match ecx.memory.alloc_map.remove(&alloc_id) {
        Some(entry) => entry,
        None => {
            // Pointer not found in local memory map. It is either a pointer to the global
            // map, or dangling.
            // If the pointer is dangling (neither in local nor global memory), we leave it
            // to validation to error. The `delay_span_bug` ensures that we don't forget such
            // a check in validation.
            if tcx.alloc_map.lock().get(alloc_id).is_none() {
                tcx.sess.delay_span_bug(ecx.tcx.span, "tried to intern dangling pointer");
            }
            // treat dangling pointers like other statics
            // just to stop trying to recurse into them
            return Ok(Some(IsStaticOrFn));
        }
    };
    // This match is just a canary for future changes to `MemoryKind`, which most likely need
    // changes in this function.
    match kind {
        MemoryKind::Stack | MemoryKind::Vtable | MemoryKind::CallerLocation => {}
    }
    // Set allocation mutability as appropriate. This is used by LLVM to put things into
    // read-only memory, and also by Miri when evluating other constants/statics that
    // access this one.
    if mode == InternMode::Static {
        // When `ty` is `None`, we assume no interior mutability.
        let frozen = ty.map_or(true, |ty| ty.is_freeze(ecx.tcx.tcx, ecx.param_env, ecx.tcx.span));
        // For statics, allocation mutability is the combination of the place mutability and
        // the type mutability.
        // The entire allocation needs to be mutable if it contains an `UnsafeCell` anywhere.
        if mutability == Mutability::Not && frozen {
            alloc.mutability = Mutability::Not;
        } else {
            // Just making sure we are not "upgrading" an immutable allocation to mutable.
            assert_eq!(alloc.mutability, Mutability::Mut);
        }
    } else {
        // We *could* be non-frozen at `ConstBase`, for constants like `Cell::new(0)`.
        // But we still intern that as immutable as the memory cannot be changed once the
        // initial value was computed.
        // Constants are never mutable.
        assert_eq!(
            mutability,
            Mutability::Not,
            "Something went very wrong: mutability requested for a constant"
        );
        alloc.mutability = Mutability::Not;
    };
    // link the alloc id to the actual allocation
    let alloc = tcx.intern_const_alloc(alloc);
    leftover_allocations.extend(alloc.relocations().iter().map(|&(_, ((), reloc))| reloc));
    tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
    Ok(None)
}

impl<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>> InternVisitor<'rt, 'mir, 'tcx, M> {
    fn intern_shallow(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
        ty: Option<Ty<'tcx>>,
    ) -> InterpResult<'tcx, Option<IsStaticOrFn>> {
        intern_shallow(self.ecx, self.leftover_allocations, self.mode, alloc_id, mutability, ty)
    }
}

impl<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>> ValueVisitor<'mir, 'tcx, M>
    for InternVisitor<'rt, 'mir, 'tcx, M>
{
    type V = MPlaceTy<'tcx>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'mir, 'tcx, M> {
        &self.ecx
    }

    fn visit_aggregate(
        &mut self,
        mplace: MPlaceTy<'tcx>,
        fields: impl Iterator<Item = InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        if let Some(def) = mplace.layout.ty.ty_adt_def() {
            if Some(def.did) == self.ecx.tcx.lang_items().unsafe_cell_type() {
                // We are crossing over an `UnsafeCell`, we can mutate again. This means that
                // References we encounter inside here are interned as pointing to mutable
                // allocations.
                let old = std::mem::replace(&mut self.mutability, Mutability::Mut);
                if !self.ignore_interior_mut_in_const_validation {
                    assert_ne!(
                        self.mode,
                        InternMode::Const,
                        "UnsafeCells are not allowed behind references in constants. This should \
                        have been prevented statically by const qualification. If this were \
                        allowed one would be able to change a constant at one use site and other \
                        use sites could observe that mutation.",
                    );
                }
                let walked = self.walk_aggregate(mplace, fields);
                self.mutability = old;
                return walked;
            }
        }
        self.walk_aggregate(mplace, fields)
    }

    fn visit_primitive(&mut self, mplace: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        // Handle Reference types, as these are the only relocations supported by const eval.
        // Raw pointers (and boxes) are handled by the `leftover_relocations` logic.
        let ty = mplace.layout.ty;
        if let ty::Ref(_, referenced_ty, mutability) = ty.kind {
            let value = self.ecx.read_immediate(mplace.into())?;
            let mplace = self.ecx.ref_to_mplace(value)?;
            // Handle trait object vtables.
            if let ty::Dynamic(..) =
                self.ecx.tcx.struct_tail_erasing_lifetimes(referenced_ty, self.ecx.param_env).kind
            {
                // Validation has already errored on an invalid vtable pointer so we can safely not
                // do anything if this is not a real pointer.
                if let Scalar::Ptr(vtable) = mplace.meta.unwrap_meta() {
                    // Explicitly choose `Immutable` here, since vtables are immutable, even
                    // if the reference of the fat pointer is mutable.
                    self.intern_shallow(vtable.alloc_id, Mutability::Not, None)?;
                } else {
                    self.ecx().tcx.sess.delay_span_bug(
                        rustc_span::DUMMY_SP,
                        "vtables pointers cannot be integer pointers",
                    );
                }
            }
            // Check if we have encountered this pointer+layout combination before.
            // Only recurse for allocation-backed pointers.
            if let Scalar::Ptr(ptr) = mplace.ptr {
                // We do not have any `frozen` logic here, because it's essentially equivalent to
                // the mutability except for the outermost item. Only `UnsafeCell` can "unfreeze",
                // and we check that in `visit_aggregate`.
                // This is not an inherent limitation, but one that we know to be true, because
                // const qualification enforces it. We can lift it in the future.
                match (self.mode, mutability) {
                    // immutable references are fine everywhere
                    (_, hir::Mutability::Not) => {}
                    // all is "good and well" in the unsoundness of `static mut`

                    // mutable references are ok in `static`. Either they are treated as immutable
                    // because they are behind an immutable one, or they are behind an `UnsafeCell`
                    // and thus ok.
                    (InternMode::Static, hir::Mutability::Mut) => {}
                    // we statically prevent `&mut T` via `const_qualif` and double check this here
                    (InternMode::ConstBase, hir::Mutability::Mut)
                    | (InternMode::Const, hir::Mutability::Mut) => match referenced_ty.kind {
                        ty::Array(_, n)
                            if n.eval_usize(self.ecx.tcx.tcx, self.ecx.param_env) == 0 => {}
                        ty::Slice(_)
                            if mplace.meta.unwrap_meta().to_machine_usize(self.ecx)? == 0 => {}
                        _ => bug!("const qualif failed to prevent mutable references"),
                    },
                }
                // Compute the mutability with which we'll start visiting the allocation. This is
                // what gets changed when we encounter an `UnsafeCell`.
                //
                // The only way a mutable reference actually works as a mutable reference is
                // by being in a `static mut` directly or behind another mutable reference.
                // If there's an immutable reference or we are inside a static, then our
                // mutable reference is equivalent to an immutable one. As an example:
                // `&&mut Foo` is semantically equivalent to `&&Foo`
                let mutability = self.mutability.and(mutability);
                // Recursing behind references changes the intern mode for constants in order to
                // cause assertions to trigger if we encounter any `UnsafeCell`s.
                let mode = match self.mode {
                    InternMode::ConstBase => InternMode::Const,
                    other => other,
                };
                match self.intern_shallow(ptr.alloc_id, mutability, Some(mplace.layout.ty))? {
                    // No need to recurse, these are interned already and statics may have
                    // cycles, so we don't want to recurse there
                    Some(IsStaticOrFn) => {}
                    // intern everything referenced by this value. The mutability is taken from the
                    // reference. It is checked above that mutable references only happen in
                    // `static mut`
                    None => self.ref_tracking.track((mplace, mutability, mode), || ()),
                }
            }
        }
        Ok(())
    }
}

pub enum InternKind {
    /// The `mutability` of the static, ignoring the type which may have interior mutability.
    Static(hir::Mutability),
    Constant,
    Promoted,
    ConstProp,
}

pub fn intern_const_alloc_recursive<M: CompileTimeMachine<'mir, 'tcx>>(
    ecx: &mut InterpCx<'mir, 'tcx, M>,
    intern_kind: InternKind,
    ret: MPlaceTy<'tcx>,
    ignore_interior_mut_in_const_validation: bool,
) -> InterpResult<'tcx> {
    let tcx = ecx.tcx;
    let (base_mutability, base_intern_mode) = match intern_kind {
        // `static mut` doesn't care about interior mutability, it's mutable anyway
        InternKind::Static(mutbl) => (mutbl, InternMode::Static),
        // FIXME: what about array lengths, array initializers?
        InternKind::Constant | InternKind::ConstProp => (Mutability::Not, InternMode::ConstBase),
        InternKind::Promoted => (Mutability::Not, InternMode::ConstBase),
    };

    // Type based interning.
    // `ref_tracking` tracks typed references we have seen and still need to crawl for
    // more typed information inside them.
    // `leftover_allocations` collects *all* allocations we see, because some might not
    // be available in a typed way. They get interned at the end.
    let mut ref_tracking = RefTracking::new((ret, base_mutability, base_intern_mode));
    let leftover_allocations = &mut FxHashSet::default();

    // start with the outermost allocation
    intern_shallow(
        ecx,
        leftover_allocations,
        base_intern_mode,
        // The outermost allocation must exist, because we allocated it with
        // `Memory::allocate`.
        ret.ptr.assert_ptr().alloc_id,
        base_mutability,
        Some(ret.layout.ty),
    )?;

    while let Some(((mplace, mutability, mode), _)) = ref_tracking.todo.pop() {
        let interned = InternVisitor {
            ref_tracking: &mut ref_tracking,
            ecx,
            mode,
            leftover_allocations,
            mutability,
            ignore_interior_mut_in_const_validation,
        }
        .visit_value(mplace);
        if let Err(error) = interned {
            // This can happen when e.g. the tag of an enum is not a valid discriminant. We do have
            // to read enum discriminants in order to find references in enum variant fields.
            if let err_unsup!(ValidationFailure(_)) = error.kind {
                let err = crate::const_eval::error_to_const_error(&ecx, error);
                match err.struct_error(
                    ecx.tcx,
                    "it is undefined behavior to use this value",
                    |mut diag| {
                        diag.note(crate::const_eval::note_on_undefined_behavior_error());
                        diag.emit();
                    },
                ) {
                    Ok(()) | Err(ErrorHandled::TooGeneric) | Err(ErrorHandled::Reported) => {}
                }
            }
        }
    }

    // Intern the rest of the allocations as mutable. These might be inside unions, padding, raw
    // pointers, ... So we can't intern them according to their type rules

    let mut todo: Vec<_> = leftover_allocations.iter().cloned().collect();
    while let Some(alloc_id) = todo.pop() {
        if let Some((_, mut alloc)) = ecx.memory.alloc_map.remove(&alloc_id) {
            // We can't call the `intern_shallow` method here, as its logic is tailored to safe
            // references and a `leftover_allocations` set (where we only have a todo-list here).
            // So we hand-roll the interning logic here again.
            match intern_kind {
                // Statics may contain mutable allocations even behind relocations.
                // Even for immutable statics it would be ok to have mutable allocations behind
                // raw pointers, e.g. for `static FOO: *const AtomicUsize = &AtomicUsize::new(42)`.
                InternKind::Static(_) => {}
                // Raw pointers in promoteds may only point to immutable things so we mark
                // everything as immutable.
                // It is UB to mutate through a raw pointer obtained via an immutable reference.
                // Since all references and pointers inside a promoted must by their very definition
                // be created from an immutable reference (and promotion also excludes interior
                // mutability), mutating through them would be UB.
                // There's no way we can check whether the user is using raw pointers correctly,
                // so all we can do is mark this as immutable here.
                InternKind::Promoted => {
                    alloc.mutability = Mutability::Not;
                }
                InternKind::Constant | InternKind::ConstProp => {
                    // If it's a constant, it *must* be immutable.
                    // We cannot have mutable memory inside a constant.
                    // We use `delay_span_bug` here, because this can be reached in the presence
                    // of fancy transmutes.
                    if alloc.mutability == Mutability::Mut {
                        // For better errors later, mark the allocation as immutable
                        // (on top of the delayed ICE).
                        alloc.mutability = Mutability::Not;
                        ecx.tcx.sess.delay_span_bug(ecx.tcx.span, "mutable allocation in constant");
                    }
                }
            }
            let alloc = tcx.intern_const_alloc(alloc);
            tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
            for &(_, ((), reloc)) in alloc.relocations().iter() {
                if leftover_allocations.insert(reloc) {
                    todo.push(reloc);
                }
            }
        } else if ecx.memory.dead_alloc_map.contains_key(&alloc_id) {
            // dangling pointer
            throw_unsup!(ValidationFailure("encountered dangling pointer in final constant".into()))
        } else if ecx.tcx.alloc_map.lock().get(alloc_id).is_none() {
            // We have hit an `AllocId` that is neither in local or global memory and isn't marked
            // as dangling by local memory.
            span_bug!(ecx.tcx.span, "encountered unknown alloc id {:?}", alloc_id);
        }
    }
    Ok(())
}
