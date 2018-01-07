use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::ty::layout::{self, LayoutOf};
use rustc::ty::subst::Substs;
use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::middle::const_val::ErrKind::{CheckMatchError, TypeckError};
use rustc::middle::const_val::{ConstEvalErr, ConstVal};
use rustc_const_eval::{lookup_const_by_id, ConstContext};
use rustc::mir::Field;
use rustc_data_structures::indexed_vec::Idx;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use rustc::mir::interpret::{EvalResult, EvalError, EvalErrorKind, GlobalId, Value, MemoryPointer, Pointer, PrimVal};
use super::{Place, EvalContext, StackPopCleanup, ValTy};

use rustc_const_math::ConstInt;

use std::fmt;
use std::error::Error;


pub fn mk_eval_cx<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, EvalContext<'a, 'tcx, CompileTimeEvaluator>> {
    debug!("mk_eval_cx: {:?}, {:?}", instance, param_env);
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::new(tcx, param_env, limits, CompileTimeEvaluator, ());
    let mir = ecx.load_mir(instance.def)?;
    // insert a stack frame so any queries have the correct substs
    ecx.push_stack_frame(
        instance,
        mir.span,
        mir,
        Place::undef(),
        StackPopCleanup::None,
    )?;
    Ok(ecx)
}

pub fn eval_body<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, (Pointer, Ty<'tcx>)> {
    debug!("eval_body: {:?}, {:?}", instance, param_env);
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::new(tcx, param_env, limits, CompileTimeEvaluator, ());
    let cid = GlobalId {
        instance,
        promoted: None,
    };

    if ecx.tcx.has_attr(instance.def_id(), "linkage") {
        return Err(ConstEvalError::NotConst("extern global".to_string()).into());
    }
    let instance_ty = instance.ty(tcx);
    if tcx.interpret_interner.borrow().get_cached(cid).is_none() {
        let mir = ecx.load_mir(instance.def)?;
        let layout = ecx.layout_of(instance_ty)?;
        assert!(!layout.is_unsized());
        let ptr = ecx.memory.allocate(
            layout.size.bytes(),
            layout.align,
            None,
        )?;
        tcx.interpret_interner.borrow_mut().cache(cid, ptr.alloc_id);
        let cleanup = StackPopCleanup::MarkStatic(Mutability::Immutable);
        let name = ty::tls::with(|tcx| tcx.item_path_str(instance.def_id()));
        trace!("const_eval: pushing stack frame for global: {}", name);
        ecx.push_stack_frame(
            instance,
            mir.span,
            mir,
            Place::from_ptr(ptr, layout.align),
            cleanup.clone(),
        )?;

        while ecx.step()? {}
    }
    let alloc = tcx.interpret_interner.borrow().get_cached(cid).expect("global not cached");
    Ok((MemoryPointer::new(alloc, 0).into(), instance_ty))
}

pub fn eval_body_as_integer<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: Instance<'tcx>,
) -> EvalResult<'tcx, ConstInt> {
    let ptr_ty = eval_body(tcx, instance, param_env);
    let (ptr, ty) = ptr_ty?;
    let ecx = mk_eval_cx(tcx, instance, param_env)?;
    let prim = match ecx.try_read_value(ptr, ecx.layout_of(ty)?.align, ty)? {
        Some(Value::ByVal(prim)) => prim.to_bytes()?,
        _ => return err!(TypeNotPrimitive(ty)),
    };
    use syntax::ast::{IntTy, UintTy};
    use rustc::ty::TypeVariants::*;
    use rustc_const_math::{ConstIsize, ConstUsize};
    Ok(match ty.sty {
        TyInt(IntTy::I8) => ConstInt::I8(prim as i128 as i8),
        TyInt(IntTy::I16) => ConstInt::I16(prim as i128 as i16),
        TyInt(IntTy::I32) => ConstInt::I32(prim as i128 as i32),
        TyInt(IntTy::I64) => ConstInt::I64(prim as i128 as i64),
        TyInt(IntTy::I128) => ConstInt::I128(prim as i128),
        TyInt(IntTy::Isize) => ConstInt::Isize(
            ConstIsize::new(prim as i128 as i64, tcx.sess.target.isize_ty)
                .expect("miri should already have errored"),
        ),
        TyUint(UintTy::U8) => ConstInt::U8(prim as u8),
        TyUint(UintTy::U16) => ConstInt::U16(prim as u16),
        TyUint(UintTy::U32) => ConstInt::U32(prim as u32),
        TyUint(UintTy::U64) => ConstInt::U64(prim as u64),
        TyUint(UintTy::U128) => ConstInt::U128(prim),
        TyUint(UintTy::Usize) => ConstInt::Usize(
            ConstUsize::new(prim as u64, tcx.sess.target.usize_ty)
                .expect("miri should already have errored"),
        ),
        _ => {
            return Err(
                ConstEvalError::NeedsRfc(
                    "evaluating anything other than isize/usize during typeck".to_string(),
                ).into(),
            )
        }
    })
}

pub struct CompileTimeEvaluator;

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalErrorKind::MachineError(Box::new(self)).into()
    }
}

#[derive(Clone, Debug)]
enum ConstEvalError {
    NeedsRfc(String),
    NotConst(String),
}

impl fmt::Display for ConstEvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(
                    f,
                    "\"{}\" needs an rfc before being allowed inside constants",
                    msg
                )
            }
            NotConst(ref msg) => write!(f, "Cannot evaluate within constants: \"{}\"", msg),
        }
    }
}

impl Error for ConstEvalError {
    fn description(&self) -> &str {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(_) => "this feature needs an rfc before being allowed inside constants",
            NotConst(_) => "this feature is not compatible with constant evaluation",
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl<'tcx> super::Machine<'tcx> for CompileTimeEvaluator {
    type MemoryData = ();
    type MemoryKinds = !;
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        _args: &[ValTy<'tcx>],
        span: Span,
        _sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        debug!("eval_fn_call: {:?}", instance);
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            return Err(
                ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into(),
            );
        }
        let mir = match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                // some simple things like `malloc` might get accepted in the future
                return Err(
                    ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                        .into(),
                );
            }
            Err(other) => return Err(other),
        };
        let (return_place, return_to_block) = match destination {
            Some((place, block)) => (place, StackPopCleanup::Goto(block)),
            None => (Place::undef(), StackPopCleanup::None),
        };

        ecx.push_stack_frame(
            instance,
            span,
            mir,
            return_place,
            return_to_block,
        )?;

        Ok(false)
    }


    fn call_intrinsic<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        _args: &[ValTy<'tcx>],
        dest: Place,
        dest_layout: layout::TyLayout<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &ecx.tcx.item_name(instance.def_id())[..];
        match intrinsic_name {
            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = ecx.layout_of(elem_ty)?.align.abi();
                let align_val = PrimVal::from_u128(elem_align as u128);
                ecx.write_primval(dest, align_val, dest_layout.ty)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = ecx.layout_of(ty)?.size.bytes() as u128;
                ecx.write_primval(dest, PrimVal::from_u128(size), dest_layout.ty)?;
            }

            name => return Err(ConstEvalError::NeedsRfc(format!("calling intrinsic `{}`", name)).into()),
        }

        ecx.goto_block(target);

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn try_ptr_op<'a>(
        _ecx: &EvalContext<'a, 'tcx, Self>,
        _bin_op: mir::BinOp,
        left: PrimVal,
        _left_ty: Ty<'tcx>,
        right: PrimVal,
        _right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        if left.is_bytes() && right.is_bytes() {
            Ok(None)
        } else {
            Err(
                ConstEvalError::NeedsRfc("Pointer arithmetic or comparison".to_string()).into(),
            )
        }
    }

    fn mark_static_initialized(m: !) -> EvalResult<'tcx> {
        m
    }

    fn box_alloc<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _ty: Ty<'tcx>,
        _dest: Place,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("Heap allocations via `box` keyword".to_string()).into(),
        )
    }

    fn global_item_with_linkage<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _mutability: Mutability,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NotConst("statics with `linkage` attribute".to_string()).into(),
        )
    }
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let (def_id, substs) = if let Some(resolved) = lookup_const_by_id(tcx, key) {
        resolved
    } else {
        return Err(ConstEvalErr {
            span: tcx.def_span(key.value.0),
            kind: TypeckError
        });
    };

    let tables = tcx.typeck_tables_of(def_id);
    let body = if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        let body_id = tcx.hir.body_owned_by(id);

        // Do match-check before building MIR
        if tcx.check_match(def_id).is_err() {
            return Err(ConstEvalErr {
                span: tcx.def_span(key.value.0),
                kind: CheckMatchError,
            });
        }

        tcx.mir_const_qualif(def_id);
        tcx.hir.body(body_id)
    } else {
        tcx.extern_const_body(def_id).body
    };

    // do not continue into miri if typeck errors occurred
    // it will fail horribly
    if tables.tainted_by_errors {
        return Err(ConstEvalErr { span: body.value.span, kind: TypeckError })
    }

    trace!("running old const eval");
    let old_result = ConstContext::new(tcx, key.param_env.and(substs), tables).eval(&body.value);
    trace!("old const eval produced {:?}", old_result);
    if tcx.sess.opts.debugging_opts.miri {
        let instance = ty::Instance::new(def_id, substs);
        trace!("const eval instance: {:?}, {:?}", instance, key.param_env);
        let miri_result = ::interpret::eval_body(tcx, instance, key.param_env);
        match (miri_result, old_result) {
            (Err(err), Ok(ok)) => {
                trace!("miri failed, ctfe returned {:?}", ok);
                tcx.sess.span_warn(
                    tcx.def_span(key.value.0),
                    "miri failed to eval, while ctfe succeeded",
                );
                let ecx = mk_eval_cx(tcx, instance, key.param_env).unwrap();
                let () = unwrap_miri(&ecx, Err(err));
                Ok(ok)
            },
            (_, Err(err)) => Err(err),
            (Ok((miri_val, miri_ty)), Ok(ctfe)) => {
                let mut ecx = mk_eval_cx(tcx, instance, key.param_env).unwrap();
                let layout = ecx.layout_of(miri_ty).unwrap();
                let miri_place = Place::from_primval_ptr(miri_val, layout.align);
                check_ctfe_against_miri(&mut ecx, miri_place, miri_ty, ctfe.val);
                Ok(ctfe)
            }
        }
    } else {
        old_result
    }
}

fn check_ctfe_against_miri<'a, 'tcx>(
    ecx: &mut EvalContext<'a, 'tcx, CompileTimeEvaluator>,
    miri_place: Place,
    miri_ty: Ty<'tcx>,
    ctfe: ConstVal<'tcx>,
) {
    use rustc::middle::const_val::ConstAggregate::*;
    use rustc_const_math::ConstFloat;
    use rustc::ty::TypeVariants::*;
    let miri_val = ValTy {
        value: ecx.read_place(miri_place).unwrap(),
        ty: miri_ty
    };
    match miri_ty.sty {
        TyInt(int_ty) => {
            let prim = get_prim(ecx, miri_val);
            let c = ConstInt::new_signed_truncating(prim as i128,
                                                    int_ty,
                                                    ecx.tcx.sess.target.isize_ty);
            let c = ConstVal::Integral(c);
            assert_eq!(c, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", c, ctfe);
        },
        TyUint(uint_ty) => {
            let prim = get_prim(ecx, miri_val);
            let c = ConstInt::new_unsigned_truncating(prim,
                                                     uint_ty,
                                                     ecx.tcx.sess.target.usize_ty);
            let c = ConstVal::Integral(c);
            assert_eq!(c, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", c, ctfe);
        },
        TyFloat(ty) => {
            let prim = get_prim(ecx, miri_val);
            let f = ConstVal::Float(ConstFloat { bits: prim, ty });
            assert_eq!(f, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", f, ctfe);
        },
        TyBool => {
            let bits = get_prim(ecx, miri_val);
            if bits > 1 {
                bug!("miri evaluated to {}, but expected a bool {:?}", bits, ctfe);
            }
            let b = ConstVal::Bool(bits == 1);
            assert_eq!(b, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", b, ctfe);
        },
        TyChar => {
            let bits = get_prim(ecx, miri_val);
            if let Some(cm) = ::std::char::from_u32(bits as u32) {
                assert_eq!(
                    ConstVal::Char(cm), ctfe,
                    "miri evaluated to {:?}, but expected {:?}", cm, ctfe,
                );
            } else {
                bug!("miri evaluated to {}, but expected a char {:?}", bits, ctfe);
            }
        },
        TyStr => {
            let value = ecx.follow_by_ref_value(miri_val.value, miri_val.ty);
            if let Ok(Value::ByValPair(PrimVal::Ptr(ptr), PrimVal::Bytes(len))) = value {
                let bytes = ecx
                    .memory
                    .read_bytes(ptr.into(), len as u64)
                    .expect("bad miri memory for str");
                if let Ok(s) = ::std::str::from_utf8(bytes) {
                    if let ConstVal::Str(s2) = ctfe {
                        assert_eq!(s, s2, "miri produced {:?}, but expected {:?}", s, s2);
                    } else {
                        bug!("miri produced {:?}, but expected {:?}", s, ctfe);
                    }
                } else {
                    bug!(
                        "miri failed to produce valid utf8 {:?}, while ctfe produced {:?}",
                        bytes,
                        ctfe,
                    );
                }
            } else {
                bug!("miri evaluated to {:?}, but expected a str {:?}", value, ctfe);
            }
        },
        TyArray(elem_ty, n) => {
            let n = n.val.to_const_int().unwrap().to_u64().unwrap();
            let vec: Vec<(ConstVal, Ty<'tcx>)> = match ctfe {
                ConstVal::ByteStr(arr) => arr.data.iter().map(|&b| {
                    (ConstVal::Integral(ConstInt::U8(b)), ecx.tcx.types.u8)
                }).collect(),
                ConstVal::Aggregate(Array(v)) => {
                    v.iter().map(|c| (c.val, c.ty)).collect()
                },
                ConstVal::Aggregate(Repeat(v, n)) => {
                    vec![(v.val, v.ty); n as usize]
                },
                _ => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            let layout = ecx.layout_of(miri_ty).unwrap();
            for (i, elem) in vec.into_iter().enumerate() {
                assert!((i as u64) < n);
                let (field_place, _) =
                    ecx.place_field(miri_place, Field::new(i), layout).unwrap();
                check_ctfe_against_miri(ecx, field_place, elem_ty, elem.0);
            }
        },
        TyTuple(..) => {
            let vec = match ctfe {
                ConstVal::Aggregate(Tuple(v)) => v,
                _ => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            let layout = ecx.layout_of(miri_ty).unwrap();
            for (i, elem) in vec.into_iter().enumerate() {
                let (field_place, _) =
                    ecx.place_field(miri_place, Field::new(i), layout).unwrap();
                check_ctfe_against_miri(ecx, field_place, elem.ty, elem.val);
            }
        },
        TyAdt(def, _) => {
            let mut miri_place = miri_place;
            let struct_variant = if def.is_enum() {
                let discr = ecx.read_discriminant_value(miri_place, miri_ty).unwrap();
                let variant = def.discriminants(ecx.tcx).position(|variant_discr| {
                    variant_discr.to_u128_unchecked() == discr
                }).expect("miri produced invalid enum discriminant");
                miri_place = ecx.place_downcast(miri_place, variant).unwrap();
                &def.variants[variant]
            } else {
                def.non_enum_variant()
            };
            let vec = match ctfe {
                ConstVal::Aggregate(Struct(v)) => v,
                ConstVal::Variant(did) => {
                    assert_eq!(struct_variant.fields.len(), 0);
                    assert_eq!(did, struct_variant.did);
                    return;
                },
                ctfe => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            let layout = ecx.layout_of(miri_ty).unwrap();
            for &(name, elem) in vec.into_iter() {
                let field = struct_variant.fields.iter().position(|f| f.name == name).unwrap();
                let (field_place, _) =
                    ecx.place_field(miri_place, Field::new(field), layout).unwrap();
                check_ctfe_against_miri(ecx, field_place, elem.ty, elem.val);
            }
        },
        TySlice(_) => bug!("miri produced a slice?"),
        // not supported by ctfe
        TyRawPtr(_) |
        TyRef(..) => {}
        TyDynamic(..) => bug!("miri produced a trait object"),
        TyClosure(..) => bug!("miri produced a closure"),
        TyGenerator(..) => bug!("miri produced a generator"),
        TyNever => bug!("miri produced a value of the never type"),
        TyProjection(_) => bug!("miri produced a projection"),
        TyAnon(..) => bug!("miri produced an impl Trait type"),
        TyParam(_) => bug!("miri produced an unmonomorphized type"),
        TyInfer(_) => bug!("miri produced an uninferred type"),
        TyError => bug!("miri produced a type error"),
        TyForeign(_) => bug!("miri produced an extern type"),
        // should be fine
        TyFnDef(..) => {}
        TyFnPtr(_) => {
            let value = ecx.value_to_primval(miri_val);
            let ptr = match value {
                Ok(PrimVal::Ptr(ptr)) => ptr,
                value => bug!("expected fn ptr, got {:?}", value),
            };
            let inst = ecx.memory.get_fn(ptr).unwrap();
            match ctfe {
                ConstVal::Function(did, substs) => {
                    let ctfe = ty::Instance::resolve(
                        ecx.tcx,
                        ecx.param_env,
                        did,
                        substs,
                    ).unwrap();
                    assert_eq!(inst, ctfe, "expected fn ptr {:?}, but got {:?}", ctfe, inst);
                },
                _ => bug!("ctfe produced {:?}, but miri produced function {:?}", ctfe, inst),
            }
        },
    }
}

fn get_prim<'a, 'tcx>(
    ecx: &mut EvalContext<'a, 'tcx, CompileTimeEvaluator>,
    val: ValTy<'tcx>,
) -> u128 {
    let res = ecx.value_to_primval(val).and_then(|prim| prim.to_bytes());
    unwrap_miri(ecx, res)
}

fn unwrap_miri<'a, 'tcx, T>(
    ecx: &EvalContext<'a, 'tcx, CompileTimeEvaluator>,
    res: Result<T, EvalError<'tcx>>,
) -> T {
    match res {
        Ok(val) => val,
        Err(mut err) => {
            ecx.report(&mut err);
            ecx.tcx.sess.abort_if_errors();
            bug!("{:#?}", err);
        }
    }
}
