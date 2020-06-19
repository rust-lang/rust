//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.

use super::validity::RefTracking;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty::{self, query::TyCtxtAt, Ty};

use rustc_ast::ast::Mutability;

use super::{AllocId, Allocation, InterpCx, MPlaceTy, Machine, MemoryKind, Scalar, ValueVisitor};

pub trait CompileTimeMachine<'mir, 'tcx> = Machine<
    'mir,
    'tcx,
    MemoryKind = !,
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
    ref_tracking: &'rt mut RefTracking<(MPlaceTy<'tcx>, InternMode)>,
    /// A list of all encountered allocations. After type-based interning, we traverse this list to
    /// also intern allocations that are only referenced by a raw pointer or inside a union.
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    /// The root kind of the value that we're looking at. This field is never mutated and only used
    /// for sanity assertions that will ICE when `const_qualif` screws up.
    mode: InternMode,
    /// This field stores whether we are *currently* inside an `UnsafeCell`. This can affect
    /// the intern mode of references we encounter.
    inside_unsafe_cell: bool,

    /// This flag is to avoid triggering UnsafeCells are not allowed behind references in constants
    /// for promoteds.
    /// It's a copy of `mir::Body`'s ignore_interior_mut_in_const_validation field
    ignore_interior_mut_in_const: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
enum InternMode {
    /// A static and its current mutability.  Below shared references inside a `static mut`,
    /// this is *immutable*, and below mutable references inside an `UnsafeCell`, this
    /// is *mutable*.
    Static(hir::Mutability),
    /// The "base value" of a const, which can have `UnsafeCell` (as in `const FOO: Cell<i32>`),
    /// but that interior mutability is simply ignored.
    ConstBase,
    /// The "inner values" of a const with references, where `UnsafeCell` is an error.
    ConstInner,
}

/// Signalling data structure to ensure we don't recurse
/// into the memory of other constants or statics
struct IsStaticOrFn;

fn mutable_memory_in_const(tcx: TyCtxtAt<'_>, kind: &str) {
    // FIXME: show this in validation instead so we can point at where in the value the error is?
    tcx.sess.span_err(tcx.span, &format!("mutable memory ({}) is not allowed in constant", kind));
}

/// Intern an allocation without looking at its children.
/// `mode` is the mode of the environment where we found this pointer.
/// `mutablity` is the mutability of the place to be interned; even if that says
/// `immutable` things might become mutable if `ty` is not frozen.
/// `ty` can be `None` if there is no potential interior mutability
/// to account for (e.g. for vtables).
fn intern_shallow<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>>(
    ecx: &'rt mut InterpCx<'mir, 'tcx, M>,
    leftover_allocations: &'rt mut FxHashSet<AllocId>,
    alloc_id: AllocId,
    mode: InternMode,
    ty: Option<Ty<'tcx>>,
) -> Option<IsStaticOrFn> {
    trace!("intern_shallow {:?} with {:?}", alloc_id, mode);
    // remove allocation
    let tcx = ecx.tcx;
    let (kind, mut alloc) = match ecx.memory.alloc_map.remove(&alloc_id) {
        Some(entry) => entry,
        None => {
            // Pointer not found in local memory map. It is either a pointer to the global
            // map, or dangling.
            // If the pointer is dangling (neither in local nor global memory), we leave it
            // to validation to error -- it has the much better error messages, pointing out where
            // in the value the dangling reference lies.
            // The `delay_span_bug` ensures that we don't forget such a check in validation.
            if tcx.get_global_alloc(alloc_id).is_none() {
                tcx.sess.delay_span_bug(ecx.tcx.span, "tried to intern dangling pointer");
            }
            // treat dangling pointers like other statics
            // just to stop trying to recurse into them
            return Some(IsStaticOrFn);
        }
    };
    // This match is just a canary for future changes to `MemoryKind`, which most likely need
    // changes in this function.
    match kind {
        MemoryKind::Stack | MemoryKind::Vtable | MemoryKind::CallerLocation => {}
    }
    // Set allocation mutability as appropriate. This is used by LLVM to put things into
    // read-only memory, and also by Miri when evaluating other globals that
    // access this one.
    if let InternMode::Static(mutability) = mode {
        // For this, we need to take into account `UnsafeCell`. When `ty` is `None`, we assume
        // no interior mutability.
        let frozen = ty.map_or(true, |ty| ty.is_freeze(*ecx.tcx, ecx.param_env, ecx.tcx.span));
        // For statics, allocation mutability is the combination of the place mutability and
        // the type mutability.
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

        // There are no sensible checks we can do here; grep for `mutable_memory_in_const` to
        // find the checks we are doing elsewhere to avoid even getting here for memory
        // that "wants" to be mutable.
        alloc.mutability = Mutability::Not;
    };
    // link the alloc id to the actual allocation
    let alloc = tcx.intern_const_alloc(alloc);
    leftover_allocations.extend(alloc.relocations().iter().map(|&(_, ((), reloc))| reloc));
    tcx.set_alloc_id_memory(alloc_id, alloc);
    None
}

impl<'rt, 'mir, 'tcx, M: CompileTimeMachine<'mir, 'tcx>> InternVisitor<'rt, 'mir, 'tcx, M> {
    fn intern_shallow(
        &mut self,
        alloc_id: AllocId,
        mode: InternMode,
        ty: Option<Ty<'tcx>>,
    ) -> Option<IsStaticOrFn> {
        intern_shallow(self.ecx, self.leftover_allocations, alloc_id, mode, ty)
    }
}

impl<'rt, 'mir, 'tcx: 'mir, M: CompileTimeMachine<'mir, 'tcx>> ValueVisitor<'mir, 'tcx, M>
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
                if self.mode == InternMode::ConstInner && !self.ignore_interior_mut_in_const {
                    // We do not actually make this memory mutable.  But in case the user
                    // *expected* it to be mutable, make sure we error.  This is just a
                    // sanity check to prevent users from accidentally exploiting the UB
                    // they caused.  It also helps us to find cases where const-checking
                    // failed to prevent an `UnsafeCell` (but as `ignore_interior_mut_in_const`
                    // shows that part is not airtight).
                    mutable_memory_in_const(self.ecx.tcx, "`UnsafeCell`");
                }
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

    fn visit_value(&mut self, mplace: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        // Handle Reference types, as these are the only relocations supported by const eval.
        // Raw pointers (and boxes) are handled by the `leftover_relocations` logic.
        let tcx = self.ecx.tcx;
        let ty = mplace.layout.ty;
        if let ty::Ref(_, referenced_ty, ref_mutability) = ty.kind {
            let value = self.ecx.read_immediate(mplace.into())?;
            let mplace = self.ecx.ref_to_mplace(value)?;
            assert_eq!(mplace.layout.ty, referenced_ty);
            // Handle trait object vtables.
            if let ty::Dynamic(..) =
                tcx.struct_tail_erasing_lifetimes(referenced_ty, self.ecx.param_env).kind
            {
                // Validation will error (with a better message) on an invalid vtable pointer
                // so we can safely not do anything if this is not a real pointer.
                if let Scalar::Ptr(vtable) = mplace.meta.unwrap_meta() {
                    // Explicitly choose const mode here, since vtables are immutable, even
                    // if the reference of the fat pointer is mutable.
                    self.intern_shallow(vtable.alloc_id, InternMode::ConstInner, None);
                } else {
                    // Let validation show the error message, but make sure it *does* error.
                    tcx.sess
                        .delay_span_bug(tcx.span, "vtables pointers cannot be integer pointers");
                }
            }
            // Check if we have encountered this pointer+layout combination before.
            // Only recurse for allocation-backed pointers.
            if let Scalar::Ptr(ptr) = mplace.ptr {
                // Compute the mode with which we intern this.
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
                                // We do *not* consier `freeze` here -- that is done more precisely
                                // when traversing the referenced data (by tracking `UnsafeCell`).
                                InternMode::Static(Mutability::Not)
                            }
                            Mutability::Mut => {
                                // Mutable reference.
                                InternMode::Static(mutbl)
                            }
                        }
                    }
                    InternMode::ConstBase | InternMode::ConstInner => {
                        // Ignore `UnsafeCell`, everything is immutable.  Do some sanity checking
                        // for mutable references that we encounter -- they must all be ZST.
                        // This helps to prevent users from accidentally exploiting UB that they
                        // caused (by somehow getting a mutable reference in a `const`).
                        if ref_mutability == Mutability::Mut {
                            match referenced_ty.kind {
                                ty::Array(_, n) if n.eval_usize(*tcx, self.ecx.param_env) == 0 => {}
                                ty::Slice(_)
                                    if mplace.meta.unwrap_meta().to_machine_usize(self.ecx)?
                                        == 0 => {}
                                _ => mutable_memory_in_const(tcx, "`&mut`"),
                            }
                        } else {
                            // A shared reference. We cannot check `freeze` here due to references
                            // like `&dyn Trait` that are actually immutable.  We do check for
                            // concrete `UnsafeCell` when traversing the pointee though (if it is
                            // a new allocation, not yet interned).
                        }
                        // Go on with the "inner" rules.
                        InternMode::ConstInner
                    }
                };
                match self.intern_shallow(ptr.alloc_id, ref_mode, Some(referenced_ty)) {
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
pub fn intern_const_alloc_recursive<M: CompileTimeMachine<'mir, 'tcx>>(
    ecx: &mut InterpCx<'mir, 'tcx, M>,
    intern_kind: InternKind,
    ret: MPlaceTy<'tcx>,
    ignore_interior_mut_in_const: bool,
) where
    'tcx: 'mir,
{
    let tcx = ecx.tcx;
    let base_intern_mode = match intern_kind {
        InternKind::Static(mutbl) => InternMode::Static(mutbl),
        // FIXME: what about array lengths, array initializers?
        InternKind::Constant | InternKind::Promoted => InternMode::ConstBase,
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
        ret.ptr.assert_ptr().alloc_id,
        base_intern_mode,
        Some(ret.layout.ty),
    );

    ref_tracking.track((ret, base_intern_mode), || ());

    while let Some(((mplace, mode), _)) = ref_tracking.todo.pop() {
        let res = InternVisitor {
            ref_tracking: &mut ref_tracking,
            ecx,
            mode,
            leftover_allocations,
            ignore_interior_mut_in_const,
            inside_unsafe_cell: false,
        }
        .visit_value(mplace);
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
                // Some errors shouldn't come up because creating them causes
                // an allocation, which we should avoid. When that happens,
                // dedicated error variants should be introduced instead.
                assert!(
                    !error.kind.allocates(),
                    "interning encountered allocating error: {}",
                    error
                );
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
            for &(_, ((), reloc)) in alloc.relocations().iter() {
                if leftover_allocations.insert(reloc) {
                    todo.push(reloc);
                }
            }
        } else if ecx.memory.dead_alloc_map.contains_key(&alloc_id) {
            // Codegen does not like dangling pointers, and generally `tcx` assumes that
            // all allocations referenced anywhere actually exist. So, make sure we error here.
            ecx.tcx.sess.span_err(ecx.tcx.span, "encountered dangling pointer in final constant");
        } else if ecx.tcx.get_global_alloc(alloc_id).is_none() {
            // We have hit an `AllocId` that is neither in local or global memory and isn't
            // marked as dangling by local memory.  That should be impossible.
            span_bug!(ecx.tcx.span, "encountered unknown alloc id {:?}", alloc_id);
        }
    }
}
