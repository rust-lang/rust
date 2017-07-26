use rustc::hir::Mutability;
use rustc::hir::Mutability::*;
use rustc::mir::{self, ValidationOp, ValidationOperand};
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::subst::Subst;
use rustc::traits::Reveal;
use rustc::infer::TransNormalize;
use rustc::middle::region::CodeExtent;

use error::{EvalError, EvalResult};
use eval_context::{EvalContext, DynamicLifetime};
use memory::{AccessKind, LockInfo};
use value::{PrimVal, Value};
use lvalue::{Lvalue, LvalueExtra};

pub type ValidationQuery<'tcx> = ValidationOperand<'tcx, Lvalue<'tcx>>;

#[derive(Copy, Clone, Debug)]
enum ValidationMode {
    Acquire,
    /// Recover because the given region ended
    Recover(CodeExtent),
    Release
}

impl ValidationMode {
    fn acquiring(self) -> bool {
        use self::ValidationMode::*;
        match self {
            Acquire | Recover(_) => true,
            Release => false,
        }
    }
}

// Validity checks
impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(crate) fn validation_op(&mut self, op: ValidationOp, operand: &ValidationOperand<'tcx, mir::Lvalue<'tcx>>) -> EvalResult<'tcx> {
        // HACK: Determine if this method is whitelisted and hence we do not perform any validation.
        {
            // The regexp we use for filtering
            use regex::Regex;
            lazy_static! {
                static ref RE: Regex = Regex::new("^(\
std::mem::swap::|\
std::mem::uninitialized::|\
std::ptr::read::|\
std::panicking::try::do_call::|\
std::slice::from_raw_parts_mut::|\
<std::heap::Heap as std::heap::Alloc>::|\
<std::mem::ManuallyDrop<T>><std::heap::AllocErr>::new$|\
<std::mem::ManuallyDrop<T> as std::ops::DerefMut><std::heap::AllocErr>::deref_mut$|\
std::sync::atomic::AtomicBool::get_mut$|\
<std::boxed::Box<T>><[a-zA-Z0-9_\\[\\]]+>::from_raw|\
<[a-zA-Z0-9_:<>]+ as std::slice::SliceIndex<[a-zA-Z0-9_\\[\\]]+>><[a-zA-Z0-9_\\[\\]]+>::get_unchecked_mut$|\
<alloc::raw_vec::RawVec<T, std::heap::Heap>><[a-zA-Z0-9_\\[\\]]+>::into_box$|\
<std::vec::Vec<T>><[a-zA-Z0-9_\\[\\]]+>::into_boxed_slice$\
)").unwrap();
            }
            // Now test
            let name = self.stack[self.cur_frame()].instance.to_string();
            if RE.is_match(&name) {
                return Ok(())
            }
        }

        // We need to monomorphize ty *without* erasing lifetimes
        let ty = operand.ty.subst(self.tcx, self.substs());
        let lval = self.eval_lvalue(&operand.lval)?;
        let query = ValidationQuery { lval, ty, re: operand.re, mutbl: operand.mutbl };

        let mode = match op {
            ValidationOp::Acquire => ValidationMode::Acquire,
            ValidationOp::Release => ValidationMode::Release,
            ValidationOp::Suspend(_) => ValidationMode::Release,
        };
        match self.validate(query.clone(), mode) {
            Err(EvalError::InvalidMemoryLockRelease { lock: LockInfo::ReadLock(_), .. }) => {
                // HACK: When &x is used while x is already borrowed read-only, AddValidation still
                // emits suspension.  This code is legit, so just ignore the error *and*
                // do NOT register a suspension.
                // TODO: Integrate AddValidation better with borrowck so that we can/ not emit
                // these wrong validation statements.  This is all pretty fragile right now.
                return Ok(());
            }
            res => res,
        }?;
        // Now that we are here, we know things went well.  Time to register the suspension.
        match op {
            ValidationOp::Suspend(ce) => {
                if query.mutbl == MutMutable {
                    let lft = DynamicLifetime { frame: self.cur_frame(), region: Some(ce) };
                    trace!("Suspending {:?} until {:?}", query, ce);
                    self.suspended.entry(lft).or_insert_with(Vec::new).push(query);
                }
            }
            _ => {}
        };
        Ok(())
    }

    pub(crate) fn end_region(&mut self, ce: CodeExtent) -> EvalResult<'tcx> {
        self.memory.locks_lifetime_ended(Some(ce));
        // Recover suspended lvals
        let lft = DynamicLifetime { frame: self.cur_frame(), region: Some(ce) };
        if let Some(queries) = self.suspended.remove(&lft) {
            for query in queries {
                trace!("Recovering {:?} from suspension", query);
                self.validate(query, ValidationMode::Recover(ce))?;
            }
        }
        Ok(())
    }

    fn validate_variant(
        &mut self,
        query: ValidationQuery<'tcx>,
        variant: &ty::VariantDef,
        subst: &ty::subst::Substs<'tcx>,
        mode: ValidationMode,
    ) -> EvalResult<'tcx> {
        // TODO: Maybe take visibility/privacy into account.
        for (idx, field) in variant.fields.iter().enumerate() {
            let field_ty = field.ty(self.tcx, subst);
            let field_lvalue = self.lvalue_field(query.lval, idx, query.ty, field_ty)?;
            self.validate(ValidationQuery { lval: field_lvalue, ty: field_ty, ..query }, mode)?;
        }
        Ok(())
    }

    fn validate_ptr(&mut self, val: Value, pointee_ty: Ty<'tcx>, re: Option<CodeExtent>, mutbl: Mutability, mode: ValidationMode) -> EvalResult<'tcx> {
        // Check alignment and non-NULLness
        let (_, align) = self.size_and_align_of_dst(pointee_ty, val)?;
        let ptr = val.into_ptr(&mut self.memory)?;
        self.memory.check_align(ptr, align)?;

        // Recurse
        let pointee_lvalue = self.val_to_lvalue(val, pointee_ty)?;
        self.validate(ValidationQuery { lval: pointee_lvalue, ty: pointee_ty, re, mutbl }, mode)
    }

    /// Validate the lvalue at the given type. If `acquire` is false, just do a release of all write locks
    #[inline]
    fn validate(&mut self, query: ValidationQuery<'tcx>, mode: ValidationMode) -> EvalResult<'tcx>
    {
        match self.try_validate(query, mode) {
            // HACK: If, during releasing, we hit memory we cannot use, we just ignore that.
            // This can happen because releases are added before drop elaboration.
            // TODO: Fix the MIR so that these releases do not happen.
            res @ Err(EvalError::DanglingPointerDeref) | res @ Err(EvalError::ReadUndefBytes) => {
                if let ValidationMode::Release = mode {
                    return Ok(());
                }
                res
            }
            res => res,
        }
    }

    fn try_validate(&mut self, mut query: ValidationQuery<'tcx>, mode: ValidationMode) -> EvalResult<'tcx>
    {
        use rustc::ty::TypeVariants::*;
        use rustc::ty::RegionKind::*;
        use rustc::ty::AdtKind;

        // No point releasing shared stuff.
        if !mode.acquiring() && query.mutbl == MutImmutable {
            return Ok(());
        }
        // When we recover, we may see data whose validity *just* ended.  Do not acquire it.
        if let ValidationMode::Recover(ce) = mode {
            if Some(ce) == query.re {
                return Ok(());
            }
        }

        // HACK: For now, bail out if we hit a dead local during recovery (can happen because sometimes we have
        // StorageDead before EndRegion).
        // TODO: We should rather fix the MIR.
        // HACK: Releasing on dead/undef local variables is a NOP.  This can happen because of releases being added
        // before drop elaboration.
        // TODO: Fix the MIR so that these releases do not happen.
        match query.lval {
            Lvalue::Local { frame, local } => {
                let res = self.stack[frame].get_local(local);
                match (res, mode) {
                    (Err(EvalError::DeadLocal), ValidationMode::Recover(_)) |
                    (Err(EvalError::DeadLocal), ValidationMode::Release) |
                    (Ok(Value::ByVal(PrimVal::Undef)), ValidationMode::Release) => {
                        return Ok(());
                    }
                    _ => {},
                }
            },
            _ => {}
        }

        // This is essentially a copy of normalize_associated_type, but without erasure
        if query.ty.has_projection_types() {
            let param_env = ty::ParamEnv::empty(Reveal::All);
            let old_ty = query.ty;
            query.ty = self.tcx.infer_ctxt().enter(move |infcx| {
                old_ty.trans_normalize(&infcx, param_env)
            })
        }
        trace!("{:?} on {:?}", mode, query);

        // Decide whether this type *owns* the memory it covers (like integers), or whether it
        // just assembles pieces (that each own their memory) together to a larger whole.
        // TODO: Currently, we don't acquire locks for padding and discriminants. We should.
        let is_owning = match query.ty.sty {
            TyInt(_) | TyUint(_) | TyRawPtr(_) |
            TyBool | TyFloat(_) | TyChar | TyStr |
            TyRef(..) | TyFnPtr(..) | TyFnDef(..) | TyNever => true,
            TyAdt(adt, _) if adt.is_box() => true,
            TySlice(_) | TyAdt(_, _) | TyTuple(..) | TyClosure(..) | TyArray(..) | TyDynamic(..) => false,
            TyParam(_) | TyInfer(_) | TyProjection(_) | TyAnon(..) | TyError => bug!("I got an incomplete/unnormalized type for validation"),
        };
        if is_owning {
            match query.lval {
                Lvalue::Ptr { ptr, extra, aligned: _ } => {
                    // Determine the size
                    // FIXME: Can we reuse size_and_align_of_dst for Lvalues?
                    let len = match self.type_size(query.ty)? {
                        Some(size) => {
                            assert_eq!(extra, LvalueExtra::None, "Got a fat ptr to a sized type");
                            size
                        }
                        None => {
                            // The only unsized typ we concider "owning" is TyStr.
                            assert_eq!(query.ty.sty, TyStr, "Found a surprising unsized owning type");
                            // The extra must be the length, in bytes.
                            match extra {
                                LvalueExtra::Length(len) => len,
                                _ => bug!("TyStr must have a length as extra"),
                            }
                        }
                    };
                    // Handle locking
                    if len > 0 {
                        let ptr = ptr.to_ptr()?;
                        let access = match query.mutbl { MutMutable => AccessKind::Write, MutImmutable => AccessKind::Read };
                        if mode.acquiring() {
                            self.memory.acquire_lock(ptr, len, query.re, access)?;
                        } else {
                            self.memory.release_write_lock(ptr, len)?;
                        }
                    }
                }
                Lvalue::Local { .. } | Lvalue::Global(..) => {
                    // These are not backed by memory, so we have nothing to do.
                }
            }
        }

        match query.ty.sty {
            TyInt(_) | TyUint(_) | TyRawPtr(_) => {
                // TODO: Make sure these are not undef.
                // We could do a bounds-check and other sanity checks on the lvalue, but it would be a bug in miri for this to ever fail.
                Ok(())
            }
            TyBool | TyFloat(_) | TyChar | TyStr => {
                // TODO: Check if these are valid bool/float/codepoint/UTF-8, respectively (and in particular, not undef).
                Ok(())
            }
            TyNever => {
                Err(EvalError::ValidationFailure(format!("The empty type is never valid.")))
            }
            TyRef(region, ty::TypeAndMut { ty: pointee_ty, mutbl }) => {
                let val = self.read_lvalue(query.lval)?;
                // Sharing restricts our context
                if mutbl == MutImmutable {
                    query.mutbl = MutImmutable;
                }
                // Inner lifetimes *outlive* outer ones, so only if we have no lifetime restriction yet,
                // we record the region of this borrow to the context.
                if query.re == None {
                    match *region {
                        ReScope(ce) => query.re = Some(ce),
                        // It is possible for us to encounter erased lifetimes here because the lifetimes in
                        // this functions' Subst will be erased.
                        _ => {},
                    }
                }
                self.validate_ptr(val, pointee_ty, query.re, query.mutbl, mode)
            }
            TyAdt(adt, _) if adt.is_box() => {
                let val = self.read_lvalue(query.lval)?;
                self.validate_ptr(val, query.ty.boxed_ty(), query.re, query.mutbl, mode)
            }
            TyFnPtr(_sig) => {
                let ptr = self.read_lvalue(query.lval)?.into_ptr(&mut self.memory)?.to_ptr()?;
                self.memory.get_fn(ptr)?;
                // TODO: Check if the signature matches (should be the same check as what terminator/mod.rs already does on call?).
                Ok(())
            }
            TyFnDef(..) => {
                // This is a zero-sized type with all relevant data sitting in the type.
                // There is nothing to validate.
                Ok(())
            }

            // Compound types
            TySlice(elem_ty) => {
                let len = match query.lval {
                    Lvalue::Ptr { extra: LvalueExtra::Length(len), .. } => len,
                    _ => bug!("acquire_valid of a TySlice given non-slice lvalue: {:?}", query.lval),
                };
                for i in 0..len {
                    let inner_lvalue = self.lvalue_index(query.lval, query.ty, i)?;
                    self.validate(ValidationQuery { lval: inner_lvalue, ty: elem_ty, ..query }, mode)?;
                }
                Ok(())
            }
            TyArray(elem_ty, len) => {
                for i in 0..len {
                    let inner_lvalue = self.lvalue_index(query.lval, query.ty, i as u64)?;
                    self.validate(ValidationQuery { lval: inner_lvalue, ty: elem_ty, ..query }, mode)?;
                }
                Ok(())
            }
            TyDynamic(_data, _region) => {
                // Check that this is a valid vtable
                let vtable = match query.lval {
                    Lvalue::Ptr { extra: LvalueExtra::Vtable(vtable), .. } => vtable,
                    _ => bug!("acquire_valid of a TyDynamic given non-trait-object lvalue: {:?}", query.lval),
                };
                self.read_size_and_align_from_vtable(vtable)?;
                // TODO: Check that the vtable contains all the function pointers we expect it to have.
                // Trait objects cannot have any operations performed
                // on them directly.  We cannot, in general, even acquire any locks as the trait object *could*
                // contain an UnsafeCell.  If we call functions to get access to data, we will validate
                // their return values.  So, it doesn't seem like there's anything else to do.
                Ok(())
            }
            TyAdt(adt, subst) => {
                if Some(adt.did) == self.tcx.lang_items.unsafe_cell_type() && query.mutbl == MutImmutable {
                    // No locks for shared unsafe cells.  Also no other validation, the only field is private anyway.
                    return Ok(());
                }

                match adt.adt_kind() {
                    AdtKind::Enum => {
                        // TODO: Can we get the discriminant without forcing an allocation?
                        let ptr = self.force_allocation(query.lval)?.to_ptr()?;
                        let discr = self.read_discriminant_value(ptr, query.ty)?;

                        // Get variant index for discriminant
                        let variant_idx = adt.discriminants(self.tcx)
                            .position(|variant_discr| variant_discr.to_u128_unchecked() == discr)
                            .ok_or(EvalError::InvalidDiscriminant)?;
                        let variant = &adt.variants[variant_idx];

                        if variant.fields.len() > 0 {
                            // Downcast to this variant, if needed
                            let lval = if adt.variants.len() > 1 {
                                self.eval_lvalue_projection(query.lval, query.ty, &mir::ProjectionElem::Downcast(adt, variant_idx))?
                            } else {
                                query.lval
                            };

                            // Recursively validate the fields
                            self.validate_variant(ValidationQuery { lval, ..query} , variant, subst, mode)
                        } else {
                            // No fields, nothing left to check.  Downcasting may fail, e.g. in case of a CEnum.
                            Ok(())
                        }
                    }
                    AdtKind::Struct => {
                        self.validate_variant(query, adt.struct_variant(), subst, mode)
                    }
                    AdtKind::Union => {
                        // No guarantees are provided for union types.
                        // TODO: Make sure that all access to union fields is unsafe; otherwise, we may have some checking to do (but what exactly?)
                        Ok(())
                    }
                }
            }
            TyTuple(ref types, _) => {
                for (idx, field_ty) in types.iter().enumerate() {
                    let field_lvalue = self.lvalue_field(query.lval, idx, query.ty, field_ty)?;
                    self.validate(ValidationQuery { lval: field_lvalue, ty: field_ty, ..query }, mode)?;
                }
                Ok(())
            }
            TyClosure(def_id, ref closure_substs) => {
                for (idx, field_ty) in closure_substs.upvar_tys(def_id, self.tcx).enumerate() {
                    let field_lvalue = self.lvalue_field(query.lval, idx, query.ty, field_ty)?;
                    self.validate(ValidationQuery { lval: field_lvalue, ty: field_ty, ..query }, mode)?;
                }
                // TODO: Check if the signature matches (should be the same check as what terminator/mod.rs already does on call?).
                // Is there other things we can/should check?  Like vtable pointers?
                Ok(())
            }
            _ => bug!("We already establishd that this is a type we support.")
        }
    }
}
