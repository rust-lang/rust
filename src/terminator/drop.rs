use rustc::hir::def_id::DefId;
use rustc::traits;
use rustc::ty::layout::Layout;
use rustc::ty::subst::{Substs, Kind};
use rustc::ty::{self, Ty};
use rustc::mir;
use syntax::codemap::Span;

use error::{EvalError, EvalResult};
use eval_context::{EvalContext, monomorphize_field_ty, StackPopCleanup};
use lvalue::{Lvalue, LvalueExtra};
use memory::Pointer;
use value::PrimVal;
use value::Value;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Creates stack frames for all drop impls. See `drop` for the actual content.
    pub fn eval_drop_impls(&mut self, drops: Vec<(DefId, Value, &'tcx Substs<'tcx>)>, span: Span) -> EvalResult<'tcx> {
        // add them to the stack in reverse order, because the impl that needs to run the last
        // is the one that needs to be at the bottom of the stack
        for (drop_def_id, self_arg, substs) in drops.into_iter().rev() {
            let mir = self.load_mir(drop_def_id)?;
            trace!("substs for drop glue: {:?}", substs);
            self.push_stack_frame(
                drop_def_id,
                span,
                mir,
                substs,
                Lvalue::from_ptr(Pointer::zst_ptr()),
                StackPopCleanup::None,
                Vec::new(),
            )?;
            let mut arg_locals = self.frame().mir.args_iter();
            let first = arg_locals.next().expect("drop impl has self arg");
            assert!(arg_locals.next().is_none(), "drop impl should have only one arg");
            let dest = self.eval_lvalue(&mir::Lvalue::Local(first))?;
            let ty = self.frame().mir.local_decls[first].ty;
            self.write_value(self_arg, dest, ty)?;
        }
        Ok(())
    }

    /// push DefIds of drop impls and their argument on the given vector
    pub fn drop(
        &mut self,
        lval: Lvalue<'tcx>,
        ty: Ty<'tcx>,
        drop: &mut Vec<(DefId, Value, &'tcx Substs<'tcx>)>,
    ) -> EvalResult<'tcx> {
        if !self.type_needs_drop(ty) {
            debug!("no need to drop {:?}", ty);
            return Ok(());
        }
        trace!("need to drop {:?} at {:?}", ty, lval);

        match ty.sty {
            // special case `Box` to deallocate the inner allocation
            ty::TyAdt(ref def, _) if def.is_box() => {
                let contents_ty = ty.boxed_ty();
                let val = self.read_lvalue(lval);
                // we are going through the read_value path, because that already does all the
                // checks for the trait object types. We'd only be repeating ourselves here.
                let val = self.follow_by_ref_value(val, ty)?;
                trace!("box dealloc on {:?}", val);
                match val {
                    Value::ByRef(_) => bug!("follow_by_ref_value can't result in ByRef"),
                    Value::ByVal(ptr) => {
                        assert!(self.type_is_sized(contents_ty));
                        let contents_ptr = ptr.to_ptr()?;
                        self.drop(Lvalue::from_ptr(contents_ptr), contents_ty, drop)?;
                    },
                    Value::ByValPair(prim_ptr, extra) => {
                        let ptr = prim_ptr.to_ptr()?;
                        let extra = match self.tcx.struct_tail(contents_ty).sty {
                            ty::TyDynamic(..) => LvalueExtra::Vtable(extra.to_ptr()?),
                            ty::TyStr | ty::TySlice(_) => LvalueExtra::Length(extra.to_u64()?),
                            _ => bug!("invalid fat pointer type: {}", ty),
                        };
                        self.drop(Lvalue::Ptr { ptr, extra }, contents_ty, drop)?;
                    },
                }
                // We cannot use Box's destructor, because it is a no-op and only exists to reduce
                // the number of hacks required in the compiler around the Box type.
                let box_free_fn = self.tcx.lang_items.box_free_fn().expect("no box_free lang item");
                let substs = self.tcx.intern_substs(&[Kind::from(contents_ty)]);
                // this is somewhat hacky, but hey, there's no representation difference between
                // pointers, `Box`es and references, so
                // #[lang = "box_free"] unsafe fn box_free<T>(ptr: *mut T)
                // is the same as
                // fn drop(&mut self) if Self is Box<T>
                drop.push((box_free_fn, val, substs));
            }

            ty::TyAdt(adt_def, substs) => {
                // FIXME: some structs are represented as ByValPair
                let mut lval = self.force_allocation(lval)?;
                let (adt_ptr, extra) = lval.to_ptr_and_extra();

                // run drop impl before the fields' drop impls
                if let Some(drop_def_id) = adt_def.destructor() {
                    let trait_ref = ty::Binder(ty::TraitRef {
                        def_id: self.tcx.lang_items.drop_trait().unwrap(),
                        substs: self.tcx.mk_substs_trait(ty, &[]),
                    });
                    let vtable = match self.fulfill_obligation(trait_ref) {
                        traits::VtableImpl(data) => data,
                        _ => bug!("dtor for {:?} is not an impl???", ty)
                    };
                    let val = match extra {
                        LvalueExtra::None => Value::ByVal(PrimVal::Ptr(adt_ptr)),
                        LvalueExtra::DowncastVariant(_) => bug!("downcast variant in drop"),
                        LvalueExtra::Length(n) => Value::ByValPair(PrimVal::Ptr(adt_ptr), PrimVal::from_u128(n as u128)),
                        LvalueExtra::Vtable(vtable) => Value::ByValPair(PrimVal::Ptr(adt_ptr), PrimVal::Ptr(vtable)),
                    };
                    drop.push((drop_def_id, val, vtable.substs));
                }

                let layout = self.type_layout(ty)?;
                let fields = match *layout {
                    Layout::Univariant { .. } => &adt_def.struct_variant().fields,
                    Layout::General { .. } => {
                        let discr_val = self.read_discriminant_value(adt_ptr, ty)? as u128;
                        let ptr = self.force_allocation(lval)?.to_ptr();
                        match adt_def.variants.iter().position(|v| discr_val == v.disr_val) {
                            Some(i) => {
                                lval = Lvalue::Ptr {
                                    ptr,
                                    extra: LvalueExtra::DowncastVariant(i),
                                };
                                &adt_def.variants[i].fields
                            },
                            None => return Err(EvalError::InvalidDiscriminant),
                        }
                    },
                    Layout::StructWrappedNullablePointer { nndiscr, .. } => {
                        let discr = self.read_discriminant_value(adt_ptr, ty)?;
                        if discr == nndiscr as u128 {
                            assert_eq!(discr as usize as u128, discr);
                            &adt_def.variants[discr as usize].fields
                        } else {
                            // FIXME: the zst variant might contain zst types that impl Drop
                            return Ok(()); // nothing to do, this is zero sized (e.g. `None`)
                        }
                    },
                    Layout::RawNullablePointer { nndiscr, .. } => {
                        let discr = self.read_discriminant_value(adt_ptr, ty)?;
                        if discr == nndiscr as u128 {
                            assert_eq!(discr as usize as u128, discr);
                            assert_eq!(adt_def.variants[discr as usize].fields.len(), 1);
                            let field_ty = &adt_def.variants[discr as usize].fields[0];
                            let field_ty = monomorphize_field_ty(self.tcx, field_ty, substs);
                            // FIXME: once read_discriminant_value works with lvalue, don't force
                            // alloc in the RawNullablePointer case
                            self.drop(lval, field_ty, drop)?;
                            return Ok(());
                        } else {
                            // FIXME: the zst variant might contain zst types that impl Drop
                            return Ok(()); // nothing to do, this is zero sized (e.g. `None`)
                        }
                    },
                    Layout::CEnum { .. } => return Ok(()),
                    _ => bug!("{:?} is not an adt layout", layout),
                };
                let tcx = self.tcx;
                self.drop_fields(
                    fields.iter().map(|field| monomorphize_field_ty(tcx, field, substs)),
                    lval,
                    ty,
                    drop,
                )?;
            }

            ty::TyTuple(fields, _) =>
                self.drop_fields(fields.into_iter().cloned(), lval, ty, drop)?,

            ty::TyDynamic(..) => {
                let (ptr, vtable) = match lval {
                    Lvalue::Ptr { ptr, extra: LvalueExtra::Vtable(vtable) } => (ptr, vtable),
                    _ => bug!("expected an lvalue with a vtable"),
                };
                let drop_fn = self.memory.read_ptr(vtable)?;
                // some values don't need to call a drop impl, so the value is null
                if drop_fn != Pointer::from_int(0) {
                    let real_ty = self.memory.get_fn(drop_fn.alloc_id)?.expect_drop_glue_real_ty()?;
                    self.drop(Lvalue::from_ptr(ptr), real_ty, drop)?;
                } else {
                    // just a sanity check
                    assert_eq!(drop_fn.offset, 0);
                }
            }

            ty::TySlice(elem_ty) => {
                let (ptr, len) = match lval {
                    Lvalue::Ptr { ptr, extra: LvalueExtra::Length(len) } => (ptr, len),
                    _ => bug!("expected an lvalue with a length"),
                };
                let size = self.type_size(elem_ty)?.expect("slice element must be sized");
                // FIXME: this creates a lot of stack frames if the element type has
                // a drop impl
                for i in 0..len {
                    self.drop(Lvalue::from_ptr(ptr.offset(i * size)), elem_ty, drop)?;
                }
            }

            ty::TyArray(elem_ty, len) => {
                let lval = self.force_allocation(lval)?;
                let (ptr, extra) = match lval {
                    Lvalue::Ptr { ptr, extra } => (ptr, extra),
                    _ => bug!("expected an lvalue with optional extra data"),
                };
                let size = self.type_size(elem_ty)?.expect("array element cannot be unsized");
                // FIXME: this creates a lot of stack frames if the element type has
                // a drop impl
                for i in 0..(len as u64) {
                    self.drop(Lvalue::Ptr { ptr: ptr.offset(i * size), extra }, elem_ty, drop)?;
                }
            }

            ty::TyClosure(def_id, substs) => {
                let fields = substs.upvar_tys(def_id, self.tcx);
                self.drop_fields(fields, lval, ty, drop)?;
            }

            _ => bug!(),
        }

        Ok(())
    }

    fn drop_fields<I>(
        &mut self,
        fields: I,
        lval: Lvalue<'tcx>,
        ty: Ty<'tcx>,
        drop: &mut Vec<(DefId, Value, &'tcx Substs<'tcx>)>,
    ) -> EvalResult<'tcx>
        where I: Iterator<Item=Ty<'tcx>>,
    {
        trace!("drop_fields: {:?} of type {}", lval, ty);
        for (i, field_ty) in fields.enumerate() {
            let field_lval = self.lvalue_field(lval, i, ty, field_ty)?;
            self.drop(field_lval, field_ty, drop)?;
        }
        Ok(())
    }

    fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_needs_drop_given_env(ty, &self.tcx.empty_parameter_environment())
    }
}
