//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.

use rustc::ty::{Ty, TyCtxt, ParamEnv, self};
use rustc::mir::interpret::{
    InterpResult, ErrorHandled,
};
use rustc::hir;
use rustc::hir::def_id::DefId;
use super::validity::RefTracking;
use rustc_data_structures::fx::FxHashSet;

use syntax::ast::Mutability;
use syntax_pos::Span;

use super::{
    ValueVisitor, MemoryKind, Pointer, AllocId, MPlaceTy, InterpError, Scalar,
};
use crate::const_eval::{CompileTimeInterpreter, CompileTimeEvalContext};

struct InternVisitor<'rt, 'mir, 'tcx> {
    /// previously encountered safe references
    ref_tracking: &'rt mut RefTracking<(MPlaceTy<'tcx>, Mutability, InternMode)>,
    ecx: &'rt mut CompileTimeEvalContext<'mir, 'tcx>,
    param_env: ParamEnv<'tcx>,
    /// The root node of the value that we're looking at. This field is never mutated and only used
    /// for sanity assertions that will ICE when `const_qualif` screws up.
    mode: InternMode,
    /// This field stores the mutability of the value *currently* being checked.
    /// It is set to mutable when an `UnsafeCell` is encountered
    /// When recursing across a reference, we don't recurse but store the
    /// value to be checked in `ref_tracking` together with the mutability at which we are checking
    /// the value.
    /// When encountering an immutable reference, we treat everything as immutable that is behind
    /// it.
    mutability: Mutability,
    /// A list of all encountered relocations. After type-based interning, we traverse this list to
    /// also intern allocations that are only referenced by a raw pointer or inside a union.
    leftover_relocations: &'rt mut FxHashSet<AllocId>,
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
enum InternMode {
    /// Mutable references must in fact be immutable due to their surrounding immutability in a
    /// `static`. In a `static mut` we start out as mutable and thus can also contain further `&mut`
    /// that will actually be treated as mutable.
    Static,
    /// UnsafeCell is OK in the value of a constant, but not behind references in a constant
    ConstBase,
    /// `UnsafeCell` ICEs
    Const,
}

/// Signalling data structure to ensure we don't recurse
/// into the memory of other constants or statics
struct IsStaticOrFn;

impl<'rt, 'mir, 'tcx> InternVisitor<'rt, 'mir, 'tcx> {
    /// Intern an allocation without looking at its children
    fn intern_shallow(
        &mut self,
        ptr: Pointer,
        mutability: Mutability,
    ) -> InterpResult<'tcx, Option<IsStaticOrFn>> {
        trace!(
            "InternVisitor::intern {:?} with {:?}",
            ptr, mutability,
        );
        // remove allocation
        let tcx = self.ecx.tcx;
        let memory = self.ecx.memory_mut();
        let (kind, mut alloc) = match memory.alloc_map.remove(&ptr.alloc_id) {
            Some(entry) => entry,
            None => {
                // if the pointer is dangling (neither in local nor global memory), we leave it
                // to validation to error. The `delay_span_bug` ensures that we don't forget such
                // a check in validation.
                if tcx.alloc_map.lock().get(ptr.alloc_id).is_none() {
                    tcx.sess.delay_span_bug(self.ecx.tcx.span, "tried to intern dangling pointer");
                }
                // treat dangling pointers like other statics
                // just to stop trying to recurse into them
                return Ok(Some(IsStaticOrFn));
            },
        };
        // This match is just a canary for future changes to `MemoryKind`, which most likely need
        // changes in this function.
        match kind {
            MemoryKind::Stack | MemoryKind::Vtable => {},
        }
        // Ensure llvm knows to only put this into immutable memory if the value is immutable either
        // by being behind a reference or by being part of a static or const without interior
        // mutability
        alloc.mutability = mutability;
        // link the alloc id to the actual allocation
        let alloc = tcx.intern_const_alloc(alloc);
        self.leftover_relocations.extend(alloc.relocations.iter().map(|&(_, ((), reloc))| reloc));
        tcx.alloc_map.lock().set_alloc_id_memory(ptr.alloc_id, alloc);
        Ok(None)
    }
}

impl<'rt, 'mir, 'tcx>
    ValueVisitor<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>
for
    InternVisitor<'rt, 'mir, 'tcx>
{
    type V = MPlaceTy<'tcx>;

    #[inline(always)]
    fn ecx(&self) -> &CompileTimeEvalContext<'mir, 'tcx> {
        &self.ecx
    }

    fn visit_aggregate(
        &mut self,
        mplace: MPlaceTy<'tcx>,
        fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        if let Some(def) = mplace.layout.ty.ty_adt_def() {
            if Some(def.did) == self.ecx.tcx.lang_items().unsafe_cell_type() {
                // We are crossing over an `UnsafeCell`, we can mutate again
                let old = std::mem::replace(&mut self.mutability, Mutability::Mutable);
                assert_ne!(
                    self.mode, InternMode::Const,
                    "UnsafeCells are not allowed behind references in constants. This should have \
                    been prevented statically by const qualification. If this were allowed one \
                    would be able to change a constant at one use site and other use sites may \
                    arbitrarily decide to change, too.",
                );
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
        if let ty::Ref(_, referenced_ty, mutability) = ty.sty {
            let value = self.ecx.read_immediate(mplace.into())?;
            // Handle trait object vtables
            if let Ok(meta) = value.to_meta() {
                if let ty::Dynamic(..) = self.ecx.tcx.struct_tail(referenced_ty).sty {
                    if let Ok(vtable) = meta.unwrap().to_ptr() {
                        // explitly choose `Immutable` here, since vtables are immutable, even
                        // if the reference of the fat pointer is mutable
                        self.intern_shallow(vtable, Mutability::Immutable)?;
                    }
                }
            }
            let mplace = self.ecx.ref_to_mplace(value)?;
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
                    (_, hir::Mutability::MutImmutable) => {},
                    // all is "good and well" in the unsoundness of `static mut`

                    // mutable references are ok in `static`. Either they are treated as immutable
                    // because they are behind an immutable one, or they are behind an `UnsafeCell`
                    // and thus ok.
                    (InternMode::Static, hir::Mutability::MutMutable) => {},
                    // we statically prevent `&mut T` via `const_qualif` and double check this here
                    (InternMode::ConstBase, hir::Mutability::MutMutable) |
                    (InternMode::Const, hir::Mutability::MutMutable) => {
                        match referenced_ty.sty {
                            ty::Array(_, n) if n.unwrap_usize(self.ecx.tcx.tcx) == 0 => {}
                            ty::Slice(_)
                                if value.to_meta().unwrap().unwrap().to_usize(self.ecx)? == 0 => {}
                            _ => bug!("const qualif failed to prevent mutable references"),
                        }
                    },
                }
                // Compute the mutability with which we'll start visiting the allocation. This is
                // what gets changed when we encounter an `UnsafeCell`
                let mutability = match (self.mutability, mutability) {
                    // The only way a mutable reference actually works as a mutable reference is
                    // by being in a `static mut` directly or behind another mutable reference.
                    // If there's an immutable reference or we are inside a static, then our
                    // mutable reference is equivalent to an immutable one. As an example:
                    // `&&mut Foo` is semantically equivalent to `&&Foo`
                    (Mutability::Mutable, hir::Mutability::MutMutable) => Mutability::Mutable,
                    _ => Mutability::Immutable,
                };
                // Compute the mutability of the allocation
                let intern_mutability = intern_mutability(
                    self.ecx.tcx.tcx,
                    self.param_env,
                    mplace.layout.ty,
                    self.ecx.tcx.span,
                    mutability,
                );
                // Recursing behind references changes the intern mode for constants in order to
                // cause assertions to trigger if we encounter any `UnsafeCell`s.
                let mode = match self.mode {
                    InternMode::ConstBase => InternMode::Const,
                    other => other,
                };
                match self.intern_shallow(ptr, intern_mutability)? {
                    // No need to recurse, these are interned already and statics may have
                    // cycles, so we don't want to recurse there
                    Some(IsStaticOrFn) => {},
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

/// Figure out the mutability of the allocation.
/// Mutable if it has interior mutability *anywhere* in the type.
fn intern_mutability<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    mutability: Mutability,
) -> Mutability {
    let has_interior_mutability = !ty.is_freeze(tcx, param_env, span);
    if has_interior_mutability {
        Mutability::Mutable
    } else {
        mutability
    }
}

pub fn intern_const_alloc_recursive(
    ecx: &mut CompileTimeEvalContext<'mir, 'tcx>,
    def_id: DefId,
    ret: MPlaceTy<'tcx>,
    // FIXME(oli-obk): can we scrap the param env? I think we can, the final value of a const eval
    // must always be monomorphic, right?
    param_env: ty::ParamEnv<'tcx>,
) -> InterpResult<'tcx> {
    let tcx = ecx.tcx;
    // this `mutability` is the mutability of the place, ignoring the type
    let (mutability, base_intern_mode) = match tcx.static_mutability(def_id) {
        Some(hir::Mutability::MutImmutable) => (Mutability::Immutable, InternMode::Static),
        None => (Mutability::Immutable, InternMode::ConstBase),
        // `static mut` doesn't care about interior mutability, it's mutable anyway
        Some(hir::Mutability::MutMutable) => (Mutability::Mutable, InternMode::Static),
    };

    // type based interning
    let mut ref_tracking = RefTracking::new((ret, mutability, base_intern_mode));
    let leftover_relocations = &mut FxHashSet::default();

    // This mutability is the combination of the place mutability and the type mutability. If either
    // is mutable, `alloc_mutability` is mutable. This exists because the entire allocation needs
    // to be mutable if it contains an `UnsafeCell` anywhere. The other `mutability` exists so that
    // the visitor does not treat everything outside the `UnsafeCell` as mutable.
    let alloc_mutability = intern_mutability(
        tcx.tcx, param_env, ret.layout.ty, tcx.span, mutability,
    );

    // start with the outermost allocation
    InternVisitor {
        ref_tracking: &mut ref_tracking,
        ecx,
        mode: base_intern_mode,
        leftover_relocations,
        param_env,
        mutability,
    }.intern_shallow(ret.ptr.to_ptr()?, alloc_mutability)?;

    while let Some(((mplace, mutability, mode), _)) = ref_tracking.todo.pop() {
        let interned = InternVisitor {
            ref_tracking: &mut ref_tracking,
            ecx,
            mode,
            leftover_relocations,
            param_env,
            mutability,
        }.visit_value(mplace);
        if let Err(error) = interned {
            // This can happen when e.g. the tag of an enum is not a valid discriminant. We do have
            // to read enum discriminants in order to find references in enum variant fields.
            if let InterpError::ValidationFailure(_) = error.kind {
                let err = crate::const_eval::error_to_const_error(&ecx, error);
                match err.struct_error(ecx.tcx, "it is undefined behavior to use this value") {
                    Ok(mut diag) => {
                        diag.note("The rules on what exactly is undefined behavior aren't clear, \
                            so this check might be overzealous. Please open an issue on the rust \
                            compiler repository if you believe it should not be considered \
                            undefined behavior",
                        );
                        diag.emit();
                    }
                    Err(ErrorHandled::TooGeneric) |
                    Err(ErrorHandled::Reported) => {},
                }
            }
        }
    }

    // Intern the rest of the allocations as mutable. These might be inside unions, padding, raw
    // pointers, ... So we can't intern them according to their type rules

    let mut todo: Vec<_> = leftover_relocations.iter().cloned().collect();
    while let Some(alloc_id) = todo.pop() {
        if let Some((_, alloc)) = ecx.memory_mut().alloc_map.remove(&alloc_id) {
            // We can't call the `intern` method here, as its logic is tailored to safe references.
            // So we hand-roll the interning logic here again
            let alloc = tcx.intern_const_alloc(alloc);
            tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
            for &(_, ((), reloc)) in alloc.relocations.iter() {
                if leftover_relocations.insert(reloc) {
                    todo.push(reloc);
                }
            }
        } else if ecx.memory().dead_alloc_map.contains_key(&alloc_id) {
            // dangling pointer
            return err!(ValidationFailure(
                "encountered dangling pointer in final constant".into(),
            ))
        }
    }
    Ok(())
}
