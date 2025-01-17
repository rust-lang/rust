use rustc_abi::{BackendRepr, Size};
use rustc_middle::mir::{Mutability, RetagKind};
use rustc_middle::ty::layout::HasTypingEnv;
use rustc_middle::ty::{self, Ty};
use rustc_span::def_id::DefId;

use crate::borrow_tracker::{GlobalState, GlobalStateInner, ProtectorKind};
use crate::concurrency::data_race::NaReadType;
use crate::*;

pub mod diagnostics;
mod foreign_access_skipping;
mod perms;
mod tree;
mod unimap;

#[cfg(test)]
mod exhaustive;

use self::perms::Permission;
pub use self::tree::Tree;

pub type AllocState = Tree;

impl<'tcx> Tree {
    /// Create a new allocation, i.e. a new tree
    pub fn new_allocation(
        id: AllocId,
        size: Size,
        state: &mut GlobalStateInner,
        _kind: MemoryKind,
        machine: &MiriMachine<'tcx>,
    ) -> Self {
        let tag = state.root_ptr_tag(id, machine); // Fresh tag for the root
        let span = machine.current_span();
        Tree::new(tag, size, span)
    }

    /// Check that an access on the entire range is permitted, and update
    /// the tree.
    pub fn before_memory_access(
        &mut self,
        access_kind: AccessKind,
        alloc_id: AllocId,
        prov: ProvenanceExtra,
        range: AllocRange,
        machine: &MiriMachine<'tcx>,
    ) -> InterpResult<'tcx> {
        trace!(
            "{} with tag {:?}: {:?}, size {}",
            access_kind,
            prov,
            interpret::Pointer::new(alloc_id, range.start),
            range.size.bytes(),
        );
        // TODO: for now we bail out on wildcard pointers. Eventually we should
        // handle them as much as we can.
        let tag = match prov {
            ProvenanceExtra::Concrete(tag) => tag,
            ProvenanceExtra::Wildcard => return interp_ok(()),
        };
        let global = machine.borrow_tracker.as_ref().unwrap();
        let span = machine.current_span();
        self.perform_access(
            tag,
            Some((range, access_kind, diagnostics::AccessCause::Explicit(access_kind))),
            global,
            alloc_id,
            span,
        )
    }

    /// Check that this pointer has permission to deallocate this range.
    pub fn before_memory_deallocation(
        &mut self,
        alloc_id: AllocId,
        prov: ProvenanceExtra,
        size: Size,
        machine: &MiriMachine<'tcx>,
    ) -> InterpResult<'tcx> {
        // TODO: for now we bail out on wildcard pointers. Eventually we should
        // handle them as much as we can.
        let tag = match prov {
            ProvenanceExtra::Concrete(tag) => tag,
            ProvenanceExtra::Wildcard => return interp_ok(()),
        };
        let global = machine.borrow_tracker.as_ref().unwrap();
        let span = machine.current_span();
        self.dealloc(tag, alloc_range(Size::ZERO, size), global, alloc_id, span)
    }

    pub fn expose_tag(&mut self, _tag: BorTag) {
        // TODO
    }

    /// A tag just lost its protector.
    ///
    /// This emits a special kind of access that is only applied
    /// to initialized locations, as a protection against other
    /// tags not having been made aware of the existence of this
    /// protector.
    pub fn release_protector(
        &mut self,
        machine: &MiriMachine<'tcx>,
        global: &GlobalState,
        tag: BorTag,
        alloc_id: AllocId, // diagnostics
    ) -> InterpResult<'tcx> {
        let span = machine.current_span();
        // `None` makes it the magic on-protector-end operation
        self.perform_access(tag, None, global, alloc_id, span)
    }
}

/// Policy for a new borrow.
#[derive(Debug, Clone, Copy)]
struct NewPermission {
    /// Optionally ignore the actual size to do a zero-size reborrow.
    /// If this is set then `dereferenceable` is not enforced.
    zero_size: bool,
    /// Which permission should the pointer start with.
    initial_state: Permission,
    /// Whether this pointer is part of the arguments of a function call.
    /// `protector` is `Some(_)` for all pointers marked `noalias`.
    protector: Option<ProtectorKind>,
}

impl<'tcx> NewPermission {
    /// Determine NewPermission of the reference from the type of the pointee.
    fn from_ref_ty(
        pointee: Ty<'tcx>,
        mutability: Mutability,
        kind: RetagKind,
        cx: &crate::MiriInterpCx<'tcx>,
    ) -> Option<Self> {
        let ty_is_freeze = pointee.is_freeze(*cx.tcx, cx.typing_env());
        let ty_is_unpin = pointee.is_unpin(*cx.tcx, cx.typing_env());
        let is_protected = kind == RetagKind::FnEntry;
        // As demonstrated by `tests/fail/tree_borrows/reservedim_spurious_write.rs`,
        // interior mutability and protectors interact poorly.
        // To eliminate the case of Protected Reserved IM we override interior mutability
        // in the case of a protected reference: protected references are always considered
        // "freeze" in their reservation phase.
        let initial_state = match mutability {
            Mutability::Mut if ty_is_unpin => Permission::new_reserved(ty_is_freeze, is_protected),
            Mutability::Not if ty_is_freeze => Permission::new_frozen(),
            // Raw pointers never enter this function so they are not handled.
            // However raw pointers are not the only pointers that take the parent
            // tag, this also happens for `!Unpin` `&mut`s and interior mutable
            // `&`s, which are excluded above.
            _ => return None,
        };

        let protector = is_protected.then_some(ProtectorKind::StrongProtector);
        Some(Self { zero_size: false, initial_state, protector })
    }

    /// Compute permission for `Box`-like type (`Box` always, and also `Unique` if enabled).
    /// These pointers allow deallocation so need a different kind of protector not handled
    /// by `from_ref_ty`.
    fn from_unique_ty(
        ty: Ty<'tcx>,
        kind: RetagKind,
        cx: &crate::MiriInterpCx<'tcx>,
        zero_size: bool,
    ) -> Option<Self> {
        let pointee = ty.builtin_deref(true).unwrap();
        pointee.is_unpin(*cx.tcx, cx.typing_env()).then_some(()).map(|()| {
            // Regular `Unpin` box, give it `noalias` but only a weak protector
            // because it is valid to deallocate it within the function.
            let ty_is_freeze = ty.is_freeze(*cx.tcx, cx.typing_env());
            let protected = kind == RetagKind::FnEntry;
            let initial_state = Permission::new_reserved(ty_is_freeze, protected);
            Self {
                zero_size,
                initial_state,
                protector: protected.then_some(ProtectorKind::WeakProtector),
            }
        })
    }
}

/// Retagging/reborrowing.
/// Policy on which permission to grant to each pointer should be left to
/// the implementation of NewPermission.
impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Returns the provenance that should be used henceforth.
    fn tb_reborrow(
        &mut self,
        place: &MPlaceTy<'tcx>, // parent tag extracted from here
        ptr_size: Size,
        new_perm: NewPermission,
        new_tag: BorTag,
    ) -> InterpResult<'tcx, Option<Provenance>> {
        let this = self.eval_context_mut();
        // Make sure the new permission makes sense as the initial permission of a fresh tag.
        assert!(new_perm.initial_state.is_initial());
        // Ensure we bail out if the pointer goes out-of-bounds (see miri#1050).
        this.check_ptr_access(place.ptr(), ptr_size, CheckInAllocMsg::InboundsTest)?;

        // It is crucial that this gets called on all code paths, to ensure we track tag creation.
        let log_creation = |this: &MiriInterpCx<'tcx>,
                            loc: Option<(AllocId, Size, ProvenanceExtra)>| // alloc_id, base_offset, orig_tag
         -> InterpResult<'tcx> {
            let global = this.machine.borrow_tracker.as_ref().unwrap().borrow();
            let ty = place.layout.ty;
            if global.tracked_pointer_tags.contains(&new_tag) {
                let kind_str = format!("initial state {} (pointee type {ty})", new_perm.initial_state);
                this.emit_diagnostic(NonHaltingDiagnostic::CreatedPointerTag(
                    new_tag.inner(),
                    Some(kind_str),
                    loc.map(|(alloc_id, base_offset, orig_tag)| (alloc_id, alloc_range(base_offset, ptr_size), orig_tag)),
                ));
            }
            drop(global); // don't hold that reference any longer than we have to
            interp_ok(())
        };

        trace!("Reborrow of size {:?}", ptr_size);
        let (alloc_id, base_offset, parent_prov) = match this.ptr_try_get_alloc_id(place.ptr(), 0) {
            Ok(data) => {
                // Unlike SB, we *do* a proper retag for size 0 if can identify the allocation.
                // After all, the pointer may be lazily initialized outside this initial range.
                data
            }
            Err(_) => {
                assert_eq!(ptr_size, Size::ZERO); // we did the deref check above, size has to be 0 here
                // This pointer doesn't come with an AllocId, so there's no
                // memory to do retagging in.
                trace!(
                    "reborrow of size 0: reference {:?} derived from {:?} (pointee {})",
                    new_tag,
                    place.ptr(),
                    place.layout.ty,
                );
                log_creation(this, None)?;
                // Keep original provenance.
                return interp_ok(place.ptr().provenance);
            }
        };
        log_creation(this, Some((alloc_id, base_offset, parent_prov)))?;

        let orig_tag = match parent_prov {
            ProvenanceExtra::Wildcard => return interp_ok(place.ptr().provenance), // TODO: handle wildcard pointers
            ProvenanceExtra::Concrete(tag) => tag,
        };

        trace!(
            "reborrow: reference {:?} derived from {:?} (pointee {}): {:?}, size {}",
            new_tag,
            orig_tag,
            place.layout.ty,
            interpret::Pointer::new(alloc_id, base_offset),
            ptr_size.bytes()
        );

        if let Some(protect) = new_perm.protector {
            // We register the protection in two different places.
            // This makes creating a protector slower, but checking whether a tag
            // is protected faster.
            this.frame_mut()
                .extra
                .borrow_tracker
                .as_mut()
                .unwrap()
                .protected_tags
                .push((alloc_id, new_tag));
            this.machine
                .borrow_tracker
                .as_mut()
                .expect("We should have borrow tracking data")
                .get_mut()
                .protected_tags
                .insert(new_tag, protect);
        }

        let alloc_kind = this.get_alloc_info(alloc_id).kind;
        if !matches!(alloc_kind, AllocKind::LiveData) {
            assert_eq!(ptr_size, Size::ZERO); // we did the deref check above, size has to be 0 here
            // There's not actually any bytes here where accesses could even be tracked.
            // Just produce the new provenance, nothing else to do.
            return interp_ok(Some(Provenance::Concrete { alloc_id, tag: new_tag }));
        }

        let span = this.machine.current_span();
        let alloc_extra = this.get_alloc_extra(alloc_id)?;
        let range = alloc_range(base_offset, ptr_size);
        let mut tree_borrows = alloc_extra.borrow_tracker_tb().borrow_mut();

        // All reborrows incur a (possibly zero-sized) read access to the parent
        tree_borrows.perform_access(
            orig_tag,
            Some((range, AccessKind::Read, diagnostics::AccessCause::Reborrow)),
            this.machine.borrow_tracker.as_ref().unwrap(),
            alloc_id,
            this.machine.current_span(),
        )?;
        // Record the parent-child pair in the tree.
        tree_borrows.new_child(
            orig_tag,
            new_tag,
            new_perm.initial_state,
            range,
            span,
            new_perm.protector.is_some(),
        )?;
        drop(tree_borrows);

        // Also inform the data race model (but only if any bytes are actually affected).
        if range.size.bytes() > 0 {
            if let Some(data_race) = alloc_extra.data_race.as_ref() {
                data_race.read(
                    alloc_id,
                    range,
                    NaReadType::Retag,
                    Some(place.layout.ty),
                    &this.machine,
                )?;
            }
        }

        interp_ok(Some(Provenance::Concrete { alloc_id, tag: new_tag }))
    }

    fn tb_retag_place(
        &mut self,
        place: &MPlaceTy<'tcx>,
        new_perm: NewPermission,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
        let this = self.eval_context_mut();

        // Determine the size of the reborrow.
        // For most types this is the entire size of the place, however
        // - when `extern type` is involved we use the size of the known prefix,
        // - if the pointer is not reborrowed (raw pointer) or if `zero_size` is set
        // then we override the size to do a zero-length reborrow.
        let reborrow_size = match new_perm {
            NewPermission { zero_size: false, .. } =>
                this.size_and_align_of_mplace(place)?
                    .map(|(size, _)| size)
                    .unwrap_or(place.layout.size),
            _ => Size::from_bytes(0),
        };
        trace!("Creating new permission: {:?} with size {:?}", new_perm, reborrow_size);

        // This new tag is not guaranteed to actually be used.
        //
        // If you run out of tags, consider the following optimization: adjust `tb_reborrow`
        // so that rather than taking as input a fresh tag and deciding whether it uses this
        // one or the parent it instead just returns whether a new tag should be created.
        // This will avoid creating tags than end up never being used.
        let new_tag = this.machine.borrow_tracker.as_mut().unwrap().get_mut().new_ptr();

        // Compute the actual reborrow.
        let new_prov = this.tb_reborrow(place, reborrow_size, new_perm, new_tag)?;

        // Adjust place.
        // (If the closure gets called, that means the old provenance was `Some`, and hence the new
        // one must also be `Some`.)
        interp_ok(place.clone().map_provenance(|_| new_prov.unwrap()))
    }

    /// Retags an individual pointer, returning the retagged version.
    fn tb_retag_reference(
        &mut self,
        val: &ImmTy<'tcx>,
        new_perm: NewPermission,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        let this = self.eval_context_mut();
        let place = this.ref_to_mplace(val)?;
        let new_place = this.tb_retag_place(&place, new_perm)?;
        interp_ok(ImmTy::from_immediate(new_place.to_ref(this), val.layout))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Retag a pointer. References are passed to `from_ref_ty` and
    /// raw pointers are never reborrowed.
    fn tb_retag_ptr_value(
        &mut self,
        kind: RetagKind,
        val: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        let this = self.eval_context_mut();
        let new_perm = match val.layout.ty.kind() {
            &ty::Ref(_, pointee, mutability) =>
                NewPermission::from_ref_ty(pointee, mutability, kind, this),
            _ => None,
        };
        if let Some(new_perm) = new_perm {
            this.tb_retag_reference(val, new_perm)
        } else {
            interp_ok(val.clone())
        }
    }

    /// Retag all pointers that are stored in this place.
    fn tb_retag_place_contents(
        &mut self,
        kind: RetagKind,
        place: &PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let options = this.machine.borrow_tracker.as_mut().unwrap().get_mut();
        let retag_fields = options.retag_fields;
        let unique_did =
            options.unique_is_unique.then(|| this.tcx.lang_items().ptr_unique()).flatten();
        let mut visitor = RetagVisitor { ecx: this, kind, retag_fields, unique_did };
        return visitor.visit_value(place);

        // The actual visitor.
        struct RetagVisitor<'ecx, 'tcx> {
            ecx: &'ecx mut MiriInterpCx<'tcx>,
            kind: RetagKind,
            retag_fields: RetagFields,
            unique_did: Option<DefId>,
        }
        impl<'ecx, 'tcx> RetagVisitor<'ecx, 'tcx> {
            #[inline(always)] // yes this helps in our benchmarks
            fn retag_ptr_inplace(
                &mut self,
                place: &PlaceTy<'tcx>,
                new_perm: Option<NewPermission>,
            ) -> InterpResult<'tcx> {
                if let Some(new_perm) = new_perm {
                    let val = self.ecx.read_immediate(&self.ecx.place_to_op(place)?)?;
                    let val = self.ecx.tb_retag_reference(&val, new_perm)?;
                    self.ecx.write_immediate(*val, place)?;
                }
                interp_ok(())
            }
        }
        impl<'ecx, 'tcx> ValueVisitor<'tcx, MiriMachine<'tcx>> for RetagVisitor<'ecx, 'tcx> {
            type V = PlaceTy<'tcx>;

            #[inline(always)]
            fn ecx(&self) -> &MiriInterpCx<'tcx> {
                self.ecx
            }

            /// Regardless of how `Unique` is handled, Boxes are always reborrowed.
            /// When `Unique` is also reborrowed, then it behaves exactly like `Box`
            /// except for the fact that `Box` has a non-zero-sized reborrow.
            fn visit_box(&mut self, box_ty: Ty<'tcx>, place: &PlaceTy<'tcx>) -> InterpResult<'tcx> {
                // Only boxes for the global allocator get any special treatment.
                if box_ty.is_box_global(*self.ecx.tcx) {
                    let new_perm = NewPermission::from_unique_ty(
                        place.layout.ty,
                        self.kind,
                        self.ecx,
                        /* zero_size */ false,
                    );
                    self.retag_ptr_inplace(place, new_perm)?;
                }
                interp_ok(())
            }

            fn visit_value(&mut self, place: &PlaceTy<'tcx>) -> InterpResult<'tcx> {
                // If this place is smaller than a pointer, we know that it can't contain any
                // pointers we need to retag, so we can stop recursion early.
                // This optimization is crucial for ZSTs, because they can contain way more fields
                // than we can ever visit.
                if place.layout.is_sized() && place.layout.size < self.ecx.pointer_size() {
                    return interp_ok(());
                }

                // Check the type of this value to see what to do with it (retag, or recurse).
                match place.layout.ty.kind() {
                    &ty::Ref(_, pointee, mutability) => {
                        let new_perm =
                            NewPermission::from_ref_ty(pointee, mutability, self.kind, self.ecx);
                        self.retag_ptr_inplace(place, new_perm)?;
                    }
                    ty::RawPtr(_, _) => {
                        // We definitely do *not* want to recurse into raw pointers -- wide raw
                        // pointers have fields, and for dyn Trait pointees those can have reference
                        // type!
                        // We also do not want to reborrow them.
                    }
                    ty::Adt(adt, _) if adt.is_box() => {
                        // Recurse for boxes, they require some tricky handling and will end up in `visit_box` above.
                        // (Yes this means we technically also recursively retag the allocator itself
                        // even if field retagging is not enabled. *shrug*)
                        self.walk_value(place)?;
                    }
                    ty::Adt(adt, _) if self.unique_did == Some(adt.did()) => {
                        let place = inner_ptr_of_unique(self.ecx, place)?;
                        let new_perm = NewPermission::from_unique_ty(
                            place.layout.ty,
                            self.kind,
                            self.ecx,
                            /* zero_size */ true,
                        );
                        self.retag_ptr_inplace(&place, new_perm)?;
                    }
                    _ => {
                        // Not a reference/pointer/box. Only recurse if configured appropriately.
                        let recurse = match self.retag_fields {
                            RetagFields::No => false,
                            RetagFields::Yes => true,
                            RetagFields::OnlyScalar => {
                                // Matching `ArgAbi::new` at the time of writing, only fields of
                                // `Scalar` and `ScalarPair` ABI are considered.
                                matches!(
                                    place.layout.backend_repr,
                                    BackendRepr::Scalar(..) | BackendRepr::ScalarPair(..)
                                )
                            }
                        };
                        if recurse {
                            self.walk_value(place)?;
                        }
                    }
                }
                interp_ok(())
            }
        }
    }

    /// Protect a place so that it cannot be used any more for the duration of the current function
    /// call.
    ///
    /// This is used to ensure soundness of in-place function argument/return passing.
    fn tb_protect_place(&mut self, place: &MPlaceTy<'tcx>) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
        let this = self.eval_context_mut();

        // Note: if we were to inline `new_reserved` below we would find out that
        // `ty_is_freeze` is eventually unused because it appears in a `ty_is_freeze || true`.
        // We are nevertheless including it here for clarity.
        let ty_is_freeze = place.layout.ty.is_freeze(*this.tcx, this.typing_env());
        // Retag it. With protection! That is the entire point.
        let new_perm = NewPermission {
            initial_state: Permission::new_reserved(ty_is_freeze, /* protected */ true),
            zero_size: false,
            protector: Some(ProtectorKind::StrongProtector),
        };
        this.tb_retag_place(place, new_perm)
    }

    /// Mark the given tag as exposed. It was found on a pointer with the given AllocId.
    fn tb_expose_tag(&self, alloc_id: AllocId, tag: BorTag) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();

        // Function pointers and dead objects don't have an alloc_extra so we ignore them.
        // This is okay because accessing them is UB anyway, no need for any Tree Borrows checks.
        // NOT using `get_alloc_extra_mut` since this might be a read-only allocation!
        let kind = this.get_alloc_info(alloc_id).kind;
        match kind {
            AllocKind::LiveData => {
                // This should have alloc_extra data, but `get_alloc_extra` can still fail
                // if converting this alloc_id from a global to a local one
                // uncovers a non-supported `extern static`.
                let alloc_extra = this.get_alloc_extra(alloc_id)?;
                trace!("Tree Borrows tag {tag:?} exposed in {alloc_id:?}");
                alloc_extra.borrow_tracker_tb().borrow_mut().expose_tag(tag);
            }
            AllocKind::Function | AllocKind::VTable | AllocKind::Dead => {
                // No tree borrows on these allocations.
            }
        }
        interp_ok(())
    }

    /// Display the tree.
    fn print_tree(&mut self, alloc_id: AllocId, show_unnamed: bool) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let alloc_extra = this.get_alloc_extra(alloc_id)?;
        let tree_borrows = alloc_extra.borrow_tracker_tb().borrow();
        let borrow_tracker = &this.machine.borrow_tracker.as_ref().unwrap().borrow();
        tree_borrows.print_tree(&borrow_tracker.protected_tags, show_unnamed)
    }

    /// Give a name to the pointer, usually the name it has in the source code (for debugging).
    /// The name given is `name` and the pointer that receives it is the `nth_parent`
    /// of `ptr` (with 0 representing `ptr` itself)
    fn tb_give_pointer_debug_name(
        &mut self,
        ptr: Pointer,
        nth_parent: u8,
        name: &str,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (tag, alloc_id) = match ptr.provenance {
            Some(Provenance::Concrete { tag, alloc_id }) => (tag, alloc_id),
            _ => {
                eprintln!("Can't give the name {name} to Wildcard pointer");
                return interp_ok(());
            }
        };
        let alloc_extra = this.get_alloc_extra(alloc_id)?;
        let mut tree_borrows = alloc_extra.borrow_tracker_tb().borrow_mut();
        tree_borrows.give_pointer_debug_name(tag, nth_parent, name)
    }
}

/// Takes a place for a `Unique` and turns it into a place with the inner raw pointer.
/// I.e. input is what you get from the visitor upon encountering an `adt` that is `Unique`,
/// and output can be used by `retag_ptr_inplace`.
fn inner_ptr_of_unique<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    place: &PlaceTy<'tcx>,
) -> InterpResult<'tcx, PlaceTy<'tcx>> {
    // Follows the same layout as `interpret/visitor.rs:walk_value` for `Box` in
    // `rustc_const_eval`, just with one fewer layer.
    // Here we have a `Unique(NonNull(*mut), PhantomData)`
    assert_eq!(place.layout.fields.count(), 2, "Unique must have exactly 2 fields");
    let (nonnull, phantom) = (ecx.project_field(place, 0)?, ecx.project_field(place, 1)?);
    assert!(
        phantom.layout.ty.ty_adt_def().is_some_and(|adt| adt.is_phantom_data()),
        "2nd field of `Unique` should be `PhantomData` but is `{:?}`",
        phantom.layout.ty,
    );
    // Now down to `NonNull(*mut)`
    assert_eq!(nonnull.layout.fields.count(), 1, "NonNull must have exactly 1 field");
    let ptr = ecx.project_field(&nonnull, 0)?;
    // Finally a plain `*mut`
    interp_ok(ptr)
}
