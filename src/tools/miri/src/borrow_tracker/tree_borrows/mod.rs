use rustc_abi::{BackendRepr, Size};
use rustc_middle::mir::{Mutability, RetagKind};
use rustc_middle::ty::layout::HasTypingEnv;
use rustc_middle::ty::{self, Ty};

use self::foreign_access_skipping::IdempotentForeignAccess;
use self::tree::LocationState;
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
    /// to accessed locations, as a protection against other
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
pub struct NewPermission {
    /// Permission for the frozen part of the range.
    freeze_perm: Permission,
    /// Whether a read access should be performed on the frozen part on a retag.
    freeze_access: bool,
    /// Permission for the non-frozen part of the range.
    nonfreeze_perm: Permission,
    /// Whether a read access should be performed on the non-frozen
    /// part on a retag.
    nonfreeze_access: bool,
    /// Permission for memory outside the range.
    outside_perm: Permission,
    /// Whether this pointer is part of the arguments of a function call.
    /// `protector` is `Some(_)` for all pointers marked `noalias`.
    protector: Option<ProtectorKind>,
}

impl<'tcx> NewPermission {
    /// Determine NewPermission of the reference/Box from the type of the pointee.
    ///
    /// A `ref_mutability` of `None` indicates a `Box` type.
    fn new(
        pointee: Ty<'tcx>,
        ref_mutability: Option<Mutability>,
        retag_kind: RetagKind,
        cx: &crate::MiriInterpCx<'tcx>,
    ) -> Option<Self> {
        let ty_is_unpin = pointee.is_unpin(*cx.tcx, cx.typing_env());
        let ty_is_freeze = pointee.is_freeze(*cx.tcx, cx.typing_env());
        let is_protected = retag_kind == RetagKind::FnEntry;

        if matches!(ref_mutability, Some(Mutability::Mut) | None if !ty_is_unpin) {
            // Mutable reference / Box to pinning type: retagging is a NOP.
            // FIXME: with `UnsafePinned`, this should do proper per-byte tracking.
            return None;
        }

        let freeze_perm = match ref_mutability {
            // Shared references are frozen.
            Some(Mutability::Not) => Permission::new_frozen(),
            // Mutable references and Boxes are reserved.
            _ => Permission::new_reserved_frz(),
        };
        let nonfreeze_perm = match ref_mutability {
            // Shared references are "transparent".
            Some(Mutability::Not) => Permission::new_cell(),
            // *Protected* mutable references and boxes are reserved without regarding for interior mutability.
            _ if is_protected => Permission::new_reserved_frz(),
            // Unprotected mutable references and boxes start in `ReservedIm`.
            _ => Permission::new_reserved_im(),
        };

        // Everything except for `Cell` gets an initial access.
        let initial_access = |perm: &Permission| !perm.is_cell();

        Some(NewPermission {
            freeze_perm,
            freeze_access: initial_access(&freeze_perm),
            nonfreeze_perm,
            nonfreeze_access: initial_access(&nonfreeze_perm),
            outside_perm: if ty_is_freeze { freeze_perm } else { nonfreeze_perm },
            protector: is_protected.then_some(if ref_mutability.is_some() {
                // Strong protector for references
                ProtectorKind::StrongProtector
            } else {
                // Weak protector for boxes
                ProtectorKind::WeakProtector
            }),
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
        // Ensure we bail out if the pointer goes out-of-bounds (see miri#1050).
        this.check_ptr_access(place.ptr(), ptr_size, CheckInAllocMsg::Dereferenceable)?;

        // It is crucial that this gets called on all code paths, to ensure we track tag creation.
        let log_creation = |this: &MiriInterpCx<'tcx>,
                            loc: Option<(AllocId, Size, ProvenanceExtra)>| // alloc_id, base_offset, orig_tag
         -> InterpResult<'tcx> {
            let global = this.machine.borrow_tracker.as_ref().unwrap().borrow();
            let ty = place.layout.ty;
            if global.tracked_pointer_tags.contains(&new_tag) {
                 let ty_is_freeze = ty.is_freeze(*this.tcx, this.typing_env());
                 let kind_str =
                     if ty_is_freeze {
                         format!("initial state {} (pointee type {ty})", new_perm.freeze_perm)
                     } else {
                         format!("initial state {}/{} outside/inside UnsafeCell (pointee type {ty})", new_perm.freeze_perm, new_perm.nonfreeze_perm)
                     };
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

        // When adding a new node, the SIFA of its parents needs to be updated, potentially across
        // the entire memory range. For the parts that are being accessed below, the access itself
        // trivially takes care of that. However, we have to do some more work to also deal with the
        // parts that are not being accessed. Specifically what we do is that we call
        // `update_last_accessed_after_retag` on the SIFA of the permission set for the part of
        // memory outside `perm_map` -- so that part is definitely taken care of. The remaining
        // concern is the part of memory that is in the range of `perms_map`, but not accessed
        // below. There we have two cases:
        // * If the type is `!Freeze`, then the non-accessed part uses `nonfreeze_perm`, so the
        //   `nonfreeze_perm` initialized parts are also fine. We enforce the `freeze_perm` parts to
        //   be accessed via the assert below, and thus everything is taken care of.
        // * If the type is `Freeze`, then `freeze_perm` is used everywhere (both inside and outside
        //   the initial range), and we update everything to have the `freeze_perm`'s SIFA, so there
        //   are no issues. (And this assert below is not actually needed in this case).
        assert!(new_perm.freeze_access);

        let protected = new_perm.protector.is_some();
        let precise_interior_mut = this
            .machine
            .borrow_tracker
            .as_mut()
            .unwrap()
            .get_mut()
            .borrow_tracker_method
            .get_tree_borrows_params()
            .precise_interior_mut;

        // Compute initial "inside" permissions.
        let loc_state = |frozen: bool| -> LocationState {
            let (perm, access) = if frozen {
                (new_perm.freeze_perm, new_perm.freeze_access)
            } else {
                (new_perm.nonfreeze_perm, new_perm.nonfreeze_access)
            };
            let sifa = perm.strongest_idempotent_foreign_access(protected);
            if access {
                LocationState::new_accessed(perm, sifa)
            } else {
                LocationState::new_non_accessed(perm, sifa)
            }
        };
        let perms_map = if !precise_interior_mut {
            // For `!Freeze` types, just pretend the entire thing is an `UnsafeCell`.
            let ty_is_freeze = place.layout.ty.is_freeze(*this.tcx, this.typing_env());
            let state = loc_state(ty_is_freeze);
            DedupRangeMap::new(ptr_size, state)
        } else {
            // The initial state will be overwritten by the visitor below.
            let mut perms_map: DedupRangeMap<LocationState> = DedupRangeMap::new(
                ptr_size,
                LocationState::new_accessed(
                    Permission::new_disabled(),
                    IdempotentForeignAccess::None,
                ),
            );
            this.visit_freeze_sensitive(place, ptr_size, |range, frozen| {
                let state = loc_state(frozen);
                for (_loc_range, loc) in perms_map.iter_mut(range.start, range.size) {
                    *loc = state;
                }
                interp_ok(())
            })?;
            perms_map
        };

        let alloc_extra = this.get_alloc_extra(alloc_id)?;
        let mut tree_borrows = alloc_extra.borrow_tracker_tb().borrow_mut();

        for (perm_range, perm) in perms_map.iter_all() {
            if perm.is_accessed() {
                // Some reborrows incur a read access to the parent.
                // Adjust range to be relative to allocation start (rather than to `place`).
                let range_in_alloc = AllocRange {
                    start: Size::from_bytes(perm_range.start) + base_offset,
                    size: Size::from_bytes(perm_range.end - perm_range.start),
                };

                tree_borrows.perform_access(
                    orig_tag,
                    Some((range_in_alloc, AccessKind::Read, diagnostics::AccessCause::Reborrow)),
                    this.machine.borrow_tracker.as_ref().unwrap(),
                    alloc_id,
                    this.machine.current_span(),
                )?;

                // Also inform the data race model (but only if any bytes are actually affected).
                if range_in_alloc.size.bytes() > 0 {
                    if let Some(data_race) = alloc_extra.data_race.as_vclocks_ref() {
                        data_race.read(
                            alloc_id,
                            range_in_alloc,
                            NaReadType::Retag,
                            Some(place.layout.ty),
                            &this.machine,
                        )?
                    }
                }
            }
        }

        // Record the parent-child pair in the tree.
        tree_borrows.new_child(
            base_offset,
            orig_tag,
            new_tag,
            perms_map,
            new_perm.outside_perm,
            protected,
            span,
        )?;
        drop(tree_borrows);

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
        // - if the pointer is not reborrowed (raw pointer) then we override the size
        //   to do a zero-length reborrow.
        let reborrow_size =
            this.size_and_align_of_val(place)?.map(|(size, _)| size).unwrap_or(place.layout.size);
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
                NewPermission::new(pointee, Some(mutability), kind, this),
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
        let mut visitor = RetagVisitor { ecx: this, kind, retag_fields };
        return visitor.visit_value(place);

        // The actual visitor.
        struct RetagVisitor<'ecx, 'tcx> {
            ecx: &'ecx mut MiriInterpCx<'tcx>,
            kind: RetagKind,
            retag_fields: RetagFields,
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
                    let pointee = place.layout.ty.builtin_deref(true).unwrap();
                    let new_perm =
                        NewPermission::new(pointee, /* not a ref */ None, self.kind, self.ecx);
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
                            NewPermission::new(pointee, Some(mutability), self.kind, self.ecx);
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

        // Retag it. With protection! That is the entire point.
        let new_perm = NewPermission {
            // Note: If we are creating a protected Reserved, which can
            // never be ReservedIM, the value of the `ty_is_freeze`
            // argument doesn't matter
            // (`ty_is_freeze || true` in `new_reserved` will always be `true`).
            freeze_perm: Permission::new_reserved_frz(),
            freeze_access: true,
            nonfreeze_perm: Permission::new_reserved_frz(),
            nonfreeze_access: true,
            outside_perm: Permission::new_reserved_frz(),
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
            AllocKind::Function | AllocKind::VTable | AllocKind::TypeId | AllocKind::Dead => {
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
