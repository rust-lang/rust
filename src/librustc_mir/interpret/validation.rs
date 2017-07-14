use rustc::hir::Mutability;
use rustc::mir::{self, ValidationOp};
use rustc::ty::{self, Ty};
use rustc::middle::region::CodeExtent;

use error::{EvalError, EvalResult};
use eval_context::{EvalContext};
use memory::AccessKind;
use value::Value;
use lvalue::{Lvalue, LvalueExtra};

// Validity checks
#[derive(Copy, Clone, Debug)]
pub struct ValidationCtx {
    op: ValidationOp,
    region: Option<CodeExtent>,
    mutbl: Mutability,
}

impl ValidationCtx {
    pub fn new(op: ValidationOp) -> Self {
        ValidationCtx {
            op, region: None, mutbl: Mutability::MutMutable,
        }
    }
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    fn validate_variant(
        &mut self,
        lvalue: Lvalue<'tcx>,
        ty: Ty<'tcx>,
        variant: &ty::VariantDef,
        subst: &ty::subst::Substs<'tcx>,
        vctx: ValidationCtx,
    ) -> EvalResult<'tcx> {
        // TODO: Take visibility/privacy into account.
        for (idx, field) in variant.fields.iter().enumerate() {
            let field_ty = field.ty(self.tcx, subst);
            let field_lvalue = self.lvalue_field(lvalue, idx, ty, field_ty)?;
            self.validate(field_lvalue, field_ty, vctx)?;
        }
        Ok(())
    }

    fn validate_ptr(&mut self, val: Value, pointee_ty: Ty<'tcx>, vctx: ValidationCtx) -> EvalResult<'tcx> {
        // Check alignment and non-NULLness
        let (_, align) = self.size_and_align_of_dst(pointee_ty, val)?;
        let ptr = val.into_ptr(&mut self.memory)?;
        self.memory.check_align(ptr, align)?;

        // Recurse
        let pointee_lvalue = self.val_to_lvalue(val, pointee_ty)?;
        self.validate(pointee_lvalue, pointee_ty, vctx)
    }

    /// Validate the lvalue at the given type. If `release` is true, just do a release of all write locks
    pub(super) fn validate(&mut self, lvalue: Lvalue<'tcx>, ty: Ty<'tcx>, mut vctx: ValidationCtx) -> EvalResult<'tcx>
    {
        use rustc::ty::TypeVariants::*;
        use rustc::ty::RegionKind::*;
        use rustc::ty::AdtKind;
        use self::Mutability::*;
        trace!("Validating {:?} at type {}, context {:?}", lvalue, ty, vctx);

        // Decide whether this type *owns* the memory it covers (like integers), or whether it
        // just assembles pieces (that each own their memory) together to a larger whole.
        // TODO: Currently, we don't acquire locks for padding and discriminants. We should.
        let is_owning = match ty.sty {
            TyInt(_) | TyUint(_) | TyRawPtr(_) |
            TyBool | TyFloat(_) | TyChar | TyStr |
            TyRef(..) | TyFnPtr(..) | TyNever => true,
            TyAdt(adt, _) if adt.is_box() => true,
            TySlice(_) | TyAdt(_, _) | TyTuple(..) | TyClosure(..) | TyArray(..) => false,
            TyParam(_) | TyInfer(_) => bug!("I got an incomplete type for validation"),
            _ => return Err(EvalError::Unimplemented(format!("Unimplemented type encountered when checking validity."))),
        };
        if is_owning {
            match lvalue {
                Lvalue::Ptr { ptr, extra, aligned: _ } => {
                    // Determine the size
                    // FIXME: Can we reuse size_and_align_of_dst for Lvalues?
                    let len = match self.type_size(ty)? {
                        Some(size) => {
                            assert_eq!(extra, LvalueExtra::None, "Got a fat ptr to a sized type");
                            size
                        }
                        None => {
                            // The only unsized typ we concider "owning" is TyStr.
                            assert_eq!(ty.sty, TyStr, "Found a surprising unsized owning type");
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
                        let access = match vctx.mutbl { MutMutable => AccessKind::Write, MutImmutable => AccessKind::Read };
                        match vctx.op {
                            ValidationOp::Acquire => self.memory.acquire_lock(ptr, len, vctx.region, access)?,
                            ValidationOp::Release => self.memory.release_write_lock_until(ptr, len, None)?,
                            ValidationOp::Suspend(region) => self.memory.release_write_lock_until(ptr, len, Some(region))?,
                        }
                    }
                }
                Lvalue::Local { .. } | Lvalue::Global(..) => {
                    // These are not backed by memory, so we have nothing to do.
                }
            }
        }

        match ty.sty {
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
                let val = self.read_lvalue(lvalue)?;
                // Sharing restricts our context
                if mutbl == MutImmutable {
                    // Actually, in case of releasing-validation, this means we are done.
                    if vctx.op != ValidationOp::Acquire {
                        return Ok(());
                    }
                    vctx.mutbl = MutImmutable;
                }
                // Inner lifetimes *outlive* outer ones, so only if we have no lifetime restriction yet,
                // we record the region of this borrow to the context.
                if vctx.region == None {
                    match *region {
                        ReScope(ce) => vctx.region = Some(ce),
                        // It is possible for us to encode erased lifetimes here because the lifetimes in
                        // this functions' Subst will be erased.
                        _ => {},
                    }
                }
                self.validate_ptr(val, pointee_ty, vctx)
            }
            TyAdt(adt, _) if adt.is_box() => {
                let val = self.read_lvalue(lvalue)?;
                self.validate_ptr(val, ty.boxed_ty(), vctx)
            }
            TyFnPtr(_sig) => {
                // TODO: The function names here could need some improvement.
                let ptr = self.read_lvalue(lvalue)?.into_ptr(&mut self.memory)?.to_ptr()?;
                self.memory.get_fn(ptr)?;
                // TODO: Check if the signature matches (should be the same check as what terminator/mod.rs already does on call?).
                Ok(())
            }

            // Compound types
            TySlice(elem_ty) => {
                let len = match lvalue {
                    Lvalue::Ptr { extra: LvalueExtra::Length(len), .. } => len,
                    _ => bug!("acquire_valid of a TySlice given non-slice lvalue: {:?}", lvalue),
                };
                for i in 0..len {
                    let inner_lvalue = self.lvalue_index(lvalue, ty, i)?;
                    self.validate(inner_lvalue, elem_ty, vctx)?;
                }
                Ok(())
            }
            TyArray(elem_ty, len) => {
                for i in 0..len {
                    let inner_lvalue = self.lvalue_index(lvalue, ty, i as u64)?;
                    self.validate(inner_lvalue, elem_ty, vctx)?;
                }
                Ok(())
            }
            TyAdt(adt, subst) => {
                if Some(adt.did) == self.tcx.lang_items.unsafe_cell_type() {
                    // No locks for unsafe cells.  Also no other validation, the only field is private anyway.
                    return Ok(());
                }

                match adt.adt_kind() {
                    AdtKind::Enum => {
                        // TODO: Can we get the discriminant without forcing an allocation?
                        let ptr = self.force_allocation(lvalue)?.to_ptr()?;
                        let discr = self.read_discriminant_value(ptr, ty)?;

                        // Get variant index for discriminant
                        let variant_idx = adt.discriminants(self.tcx)
                            .position(|variant_discr| variant_discr.to_u128_unchecked() == discr)
                            .ok_or(EvalError::InvalidDiscriminant)?;
                        let variant = &adt.variants[variant_idx];

                        if variant.fields.len() > 0 {
                            // Downcast to this variant, if needed
                            let lvalue = if adt.variants.len() > 1 {
                                self.eval_lvalue_projection(lvalue, ty, &mir::ProjectionElem::Downcast(adt, variant_idx))?
                            } else {
                                lvalue
                            };

                            // Recursively validate the fields
                            self.validate_variant(lvalue, ty, variant, subst, vctx)
                        } else {
                            // No fields, nothing left to check.  Downcasting may fail, e.g. in case of a CEnum.
                            Ok(())
                        }
                    }
                    AdtKind::Struct => {
                        self.validate_variant(lvalue, ty, adt.struct_variant(), subst, vctx)
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
                    let field_lvalue = self.lvalue_field(lvalue, idx, ty, field_ty)?;
                    self.validate(field_lvalue, field_ty, vctx)?;
                }
                Ok(())
            }
            TyClosure(def_id, ref closure_substs) => {
                for (idx, field_ty) in closure_substs.upvar_tys(def_id, self.tcx).enumerate() {
                    let field_lvalue = self.lvalue_field(lvalue, idx, ty, field_ty)?;
                    self.validate(field_lvalue, field_ty, vctx)?;
                }
                // TODO: Check if the signature matches (should be the same check as what terminator/mod.rs already does on call?).
                // Is there other things we can/should check?  Like vtable pointers?
                Ok(())
            }
            _ => bug!("We already establishd that this is a type we support.")
        }
    }
}
