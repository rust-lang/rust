// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::const_val::ConstVal::*;
use rustc::middle::const_val::ErrKind::*;
use rustc::middle::const_val::{ConstVal, ConstEvalErr, EvalResult, ErrKind};

use rustc::hir::map as hir_map;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits;
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::ty::util::IntTypeExt;
use rustc::ty::subst::{Substs, Subst};
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::DefIdMap;

use syntax::abi::Abi;
use syntax::ast;
use syntax::attr;
use rustc::hir::{self, Expr};
use syntax_pos::Span;

use std::cmp::Ordering;

use rustc_const_math::*;

macro_rules! signal {
    ($e:expr, $exn:expr) => {
        return Err(ConstEvalErr { span: $e.span, kind: $exn })
    }
}

macro_rules! math {
    ($e:expr, $op:expr) => {
        match $op {
            Ok(val) => val,
            Err(e) => signal!($e, ErrKind::from(e)),
        }
    }
}

/// * `DefId` is the id of the constant.
/// * `Substs` is the monomorphized substitutions for the expression.
pub fn lookup_const_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                                    -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let (def_id, _) = key.value;
    if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
        match tcx.hir.find(node_id) {
            Some(hir_map::NodeTraitItem(_)) => {
                // If we have a trait item and the substitutions for it,
                // `resolve_trait_associated_const` will select an impl
                // or the default.
                resolve_trait_associated_const(tcx, key)
            }
            _ => Some(key.value)
        }
    } else {
        match tcx.describe_def(def_id) {
            Some(Def::AssociatedConst(_)) => {
                // As mentioned in the comments above for in-crate
                // constants, we only try to find the expression for a
                // trait-associated const if the caller gives us the
                // substitutions for the reference to it.
                if tcx.trait_of_item(def_id).is_some() {
                    resolve_trait_associated_const(tcx, key)
                } else {
                    Some(key.value)
                }
            }
            _ => Some(key.value)
        }
    }
}

pub struct ConstContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    substs: &'tcx Substs<'tcx>,
    fn_args: Option<DefIdMap<ConstVal<'tcx>>>
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               param_env_and_substs: ty::ParamEnvAnd<'tcx, &'tcx Substs<'tcx>>,
               tables: &'a ty::TypeckTables<'tcx>)
               -> Self {
        ConstContext {
            tcx,
            param_env: param_env_and_substs.param_env,
            tables,
            substs: param_env_and_substs.value,
            fn_args: None
        }
    }

    /// Evaluate a constant expression in a context where the expression isn't
    /// guaranteed to be evaluatable.
    pub fn eval(&self, e: &Expr) -> EvalResult<'tcx> {
        if self.tables.tainted_by_errors {
            signal!(e, TypeckError);
        }
        eval_const_expr_partial(self, e)
    }
}

type CastResult<'tcx> = Result<ConstVal<'tcx>, ErrKind<'tcx>>;

fn eval_const_expr_partial<'a, 'tcx>(cx: &ConstContext<'a, 'tcx>,
                                     e: &Expr) -> EvalResult<'tcx> {
    let tcx = cx.tcx;
    let ety = cx.tables.expr_ty(e).subst(tcx, cx.substs);

    let result = match e.node {
      hir::ExprUnary(hir::UnNeg, ref inner) => {
        // unary neg literals already got their sign during creation
        if let hir::ExprLit(ref lit) = inner.node {
            use syntax::ast::*;
            use syntax::ast::LitIntType::*;
            const I8_OVERFLOW: u128 = i8::min_value() as u8 as u128;
            const I16_OVERFLOW: u128 = i16::min_value() as u16 as u128;
            const I32_OVERFLOW: u128 = i32::min_value() as u32 as u128;
            const I64_OVERFLOW: u128 = i64::min_value() as u64 as u128;
            const I128_OVERFLOW: u128 = i128::min_value() as u128;
            match (&lit.node, &ety.sty) {
                (&LitKind::Int(I8_OVERFLOW, _), &ty::TyInt(IntTy::I8)) |
                (&LitKind::Int(I8_OVERFLOW, Signed(IntTy::I8)), _) => {
                    return Ok(Integral(I8(i8::min_value())))
                },
                (&LitKind::Int(I16_OVERFLOW, _), &ty::TyInt(IntTy::I16)) |
                (&LitKind::Int(I16_OVERFLOW, Signed(IntTy::I16)), _) => {
                    return Ok(Integral(I16(i16::min_value())))
                },
                (&LitKind::Int(I32_OVERFLOW, _), &ty::TyInt(IntTy::I32)) |
                (&LitKind::Int(I32_OVERFLOW, Signed(IntTy::I32)), _) => {
                    return Ok(Integral(I32(i32::min_value())))
                },
                (&LitKind::Int(I64_OVERFLOW, _), &ty::TyInt(IntTy::I64)) |
                (&LitKind::Int(I64_OVERFLOW, Signed(IntTy::I64)), _) => {
                    return Ok(Integral(I64(i64::min_value())))
                },
                (&LitKind::Int(I128_OVERFLOW, _), &ty::TyInt(IntTy::I128)) |
                (&LitKind::Int(I128_OVERFLOW, Signed(IntTy::I128)), _) => {
                    return Ok(Integral(I128(i128::min_value())))
                },
                (&LitKind::Int(n, _), &ty::TyInt(IntTy::Is)) |
                (&LitKind::Int(n, Signed(IntTy::Is)), _) => {
                    match tcx.sess.target.int_type {
                        IntTy::I16 => if n == I16_OVERFLOW {
                            return Ok(Integral(Isize(Is16(i16::min_value()))));
                        },
                        IntTy::I32 => if n == I32_OVERFLOW {
                            return Ok(Integral(Isize(Is32(i32::min_value()))));
                        },
                        IntTy::I64 => if n == I64_OVERFLOW {
                            return Ok(Integral(Isize(Is64(i64::min_value()))));
                        },
                        _ => span_bug!(e.span, "typeck error")
                    }
                },
                _ => {},
            }
        }
        match cx.eval(inner)? {
          Float(f) => Float(-f),
          Integral(i) => Integral(math!(e, -i)),
          const_val => signal!(e, NegateOn(const_val)),
        }
      }
      hir::ExprUnary(hir::UnNot, ref inner) => {
        match cx.eval(inner)? {
          Integral(i) => Integral(math!(e, !i)),
          Bool(b) => Bool(!b),
          const_val => signal!(e, NotOn(const_val)),
        }
      }
      hir::ExprUnary(hir::UnDeref, _) => signal!(e, UnimplementedConstVal("deref operation")),
      hir::ExprBinary(op, ref a, ref b) => {
        // technically, if we don't have type hints, but integral eval
        // gives us a type through a type-suffix, cast or const def type
        // we need to re-eval the other value of the BinOp if it was
        // not inferred
        match (cx.eval(a)?, cx.eval(b)?) {
          (Float(a), Float(b)) => {
            use std::cmp::Ordering::*;
            match op.node {
              hir::BiAdd => Float(math!(e, a + b)),
              hir::BiSub => Float(math!(e, a - b)),
              hir::BiMul => Float(math!(e, a * b)),
              hir::BiDiv => Float(math!(e, a / b)),
              hir::BiRem => Float(math!(e, a % b)),
              hir::BiEq => Bool(math!(e, a.try_cmp(b)) == Equal),
              hir::BiLt => Bool(math!(e, a.try_cmp(b)) == Less),
              hir::BiLe => Bool(math!(e, a.try_cmp(b)) != Greater),
              hir::BiNe => Bool(math!(e, a.try_cmp(b)) != Equal),
              hir::BiGe => Bool(math!(e, a.try_cmp(b)) != Less),
              hir::BiGt => Bool(math!(e, a.try_cmp(b)) == Greater),
              _ => span_bug!(e.span, "typeck error"),
            }
          }
          (Integral(a), Integral(b)) => {
            use std::cmp::Ordering::*;
            match op.node {
              hir::BiAdd => Integral(math!(e, a + b)),
              hir::BiSub => Integral(math!(e, a - b)),
              hir::BiMul => Integral(math!(e, a * b)),
              hir::BiDiv => Integral(math!(e, a / b)),
              hir::BiRem => Integral(math!(e, a % b)),
              hir::BiBitAnd => Integral(math!(e, a & b)),
              hir::BiBitOr => Integral(math!(e, a | b)),
              hir::BiBitXor => Integral(math!(e, a ^ b)),
              hir::BiShl => Integral(math!(e, a << b)),
              hir::BiShr => Integral(math!(e, a >> b)),
              hir::BiEq => Bool(math!(e, a.try_cmp(b)) == Equal),
              hir::BiLt => Bool(math!(e, a.try_cmp(b)) == Less),
              hir::BiLe => Bool(math!(e, a.try_cmp(b)) != Greater),
              hir::BiNe => Bool(math!(e, a.try_cmp(b)) != Equal),
              hir::BiGe => Bool(math!(e, a.try_cmp(b)) != Less),
              hir::BiGt => Bool(math!(e, a.try_cmp(b)) == Greater),
              _ => span_bug!(e.span, "typeck error"),
            }
          }
          (Bool(a), Bool(b)) => {
            Bool(match op.node {
              hir::BiAnd => a && b,
              hir::BiOr => a || b,
              hir::BiBitXor => a ^ b,
              hir::BiBitAnd => a & b,
              hir::BiBitOr => a | b,
              hir::BiEq => a == b,
              hir::BiNe => a != b,
              hir::BiLt => a < b,
              hir::BiLe => a <= b,
              hir::BiGe => a >= b,
              hir::BiGt => a > b,
              _ => span_bug!(e.span, "typeck error"),
             })
          }
          (Char(a), Char(b)) => {
            Bool(match op.node {
              hir::BiEq => a == b,
              hir::BiNe => a != b,
              hir::BiLt => a < b,
              hir::BiLe => a <= b,
              hir::BiGe => a >= b,
              hir::BiGt => a > b,
              _ => span_bug!(e.span, "typeck error"),
             })
          }

          _ => signal!(e, MiscBinaryOp),
        }
      }
      hir::ExprCast(ref base, _) => {
        let base_val = cx.eval(base)?;
        let base_ty = cx.tables.expr_ty(base).subst(tcx, cx.substs);
        if ety == base_ty {
            base_val
        } else {
            match cast_const(tcx, base_val, ety) {
                Ok(val) => val,
                Err(kind) => signal!(e, kind),
            }
        }
      }
      hir::ExprPath(ref qpath) => {
        let substs = cx.tables.node_substs(e.id).subst(tcx, cx.substs);
          match cx.tables.qpath_def(qpath, e.id) {
              Def::Const(def_id) |
              Def::AssociatedConst(def_id) => {
                    match tcx.at(e.span).const_eval(cx.param_env.and((def_id, substs))) {
                        Ok(val) => val,
                        Err(ConstEvalErr { kind: TypeckError, .. }) => {
                            signal!(e, TypeckError);
                        }
                        Err(err) => {
                            debug!("bad reference: {:?}, {:?}", err.description(), err.span);
                            signal!(e, ErroneousReferencedConstant(box err))
                        },
                    }
              },
              Def::VariantCtor(variant_def, CtorKind::Const) => {
                Variant(variant_def)
              }
              Def::VariantCtor(_, CtorKind::Fn) => {
                  signal!(e, UnimplementedConstVal("enum variants"));
              }
              Def::StructCtor(_, CtorKind::Const) => {
                  ConstVal::Struct(Default::default())
              }
              Def::StructCtor(_, CtorKind::Fn) => {
                  signal!(e, UnimplementedConstVal("tuple struct constructors"))
              }
              Def::Local(def_id) => {
                  debug!("Def::Local({:?}): {:?}", def_id, cx.fn_args);
                  if let Some(val) = cx.fn_args.as_ref().and_then(|args| args.get(&def_id)) {
                      val.clone()
                  } else {
                      signal!(e, NonConstPath);
                  }
              },
              Def::Method(id) | Def::Fn(id) => Function(id, substs),
              Def::Err => span_bug!(e.span, "typeck error"),
              _ => signal!(e, NonConstPath),
          }
      }
      hir::ExprCall(ref callee, ref args) => {
          let (def_id, substs) = match cx.eval(callee)? {
              Function(def_id, substs) => (def_id, substs),
              _ => signal!(e, TypeckError),
          };

          if tcx.fn_sig(def_id).abi() == Abi::RustIntrinsic {
            let layout_of = |ty: Ty<'tcx>| {
                ty.layout(tcx, cx.param_env).map_err(|err| {
                    ConstEvalErr { span: e.span, kind: LayoutError(err) }
                })
            };
            match &tcx.item_name(def_id).as_str()[..] {
                "size_of" => {
                    let size = layout_of(substs.type_at(0))?.size(tcx);
                    return Ok(Integral(Usize(ConstUsize::new(size.bytes(),
                        tcx.sess.target.uint_type).unwrap())));
                }
                "min_align_of" => {
                    let align = layout_of(substs.type_at(0))?.align(tcx);
                    return Ok(Integral(Usize(ConstUsize::new(align.abi(),
                        tcx.sess.target.uint_type).unwrap())));
                }
                _ => signal!(e, TypeckError)
            }
          }

          let body = if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
            if let Some(fn_like) = FnLikeNode::from_node(tcx.hir.get(node_id)) {
                if fn_like.constness() == hir::Constness::Const {
                    tcx.hir.body(fn_like.body())
                } else {
                    signal!(e, TypeckError)
                }
            } else {
                signal!(e, TypeckError)
            }
          } else {
            if tcx.is_const_fn(def_id) {
                tcx.sess.cstore.item_body(tcx, def_id)
            } else {
                signal!(e, TypeckError)
            }
          };

          let arg_defs = body.arguments.iter().map(|arg| match arg.pat.node {
               hir::PatKind::Binding(_, def_id, _, _) => Some(def_id),
               _ => None
           }).collect::<Vec<_>>();
          assert_eq!(arg_defs.len(), args.len());

          let mut call_args = DefIdMap();
          for (arg, arg_expr) in arg_defs.into_iter().zip(args.iter()) {
              let arg_val = cx.eval(arg_expr)?;
              debug!("const call arg: {:?}", arg);
              if let Some(def_id) = arg {
                assert!(call_args.insert(def_id, arg_val).is_none());
              }
          }
          debug!("const call({:?})", call_args);
          let callee_cx = ConstContext {
            tcx,
            param_env: cx.param_env,
            tables: tcx.typeck_tables_of(def_id),
            substs: substs,
            fn_args: Some(call_args)
          };
          callee_cx.eval(&body.value)?
      },
      hir::ExprLit(ref lit) => match lit_to_const(&lit.node, tcx, ety) {
          Ok(val) => val,
          Err(err) => signal!(e, err),
      },
      hir::ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => cx.eval(expr)?,
            None => Tuple(vec![]),
        }
      }
      hir::ExprType(ref e, _) => cx.eval(e)?,
      hir::ExprTup(ref fields) => {
        Tuple(fields.iter().map(|e| cx.eval(e)).collect::<Result<_, _>>()?)
      }
      hir::ExprStruct(_, ref fields, _) => {
        Struct(fields.iter().map(|f| {
            cx.eval(&f.expr).map(|v| (f.name.node, v))
        }).collect::<Result<_, _>>()?)
      }
      hir::ExprIndex(ref arr, ref idx) => {
        if !tcx.sess.features.borrow().const_indexing {
            signal!(e, IndexOpFeatureGated);
        }
        let arr = cx.eval(arr)?;
        let idx = match cx.eval(idx)? {
            Integral(Usize(i)) => i.as_u64(tcx.sess.target.uint_type),
            _ => signal!(idx, IndexNotUsize),
        };
        assert_eq!(idx as usize as u64, idx);
        match arr {
            Array(ref v) => {
                if let Some(elem) = v.get(idx as usize) {
                    elem.clone()
                } else {
                    let n = v.len() as u64;
                    assert_eq!(n as usize as u64, n);
                    signal!(e, IndexOutOfBounds { len: n, index: idx })
                }
            }

            Repeat(.., n) if idx >= n => {
                signal!(e, IndexOutOfBounds { len: n, index: idx })
            }
            Repeat(ref elem, _) => (**elem).clone(),

            ByteStr(ref data) if idx >= data.len() as u64 => {
                signal!(e, IndexOutOfBounds { len: data.len() as u64, index: idx })
            }
            ByteStr(data) => {
                Integral(U8(data[idx as usize]))
            },

            _ => signal!(e, IndexedNonVec),
        }
      }
      hir::ExprArray(ref v) => {
        Array(v.iter().map(|e| cx.eval(e)).collect::<Result<_, _>>()?)
      }
      hir::ExprRepeat(ref elem, _) => {
          let n = match ety.sty {
            ty::TyArray(_, n) => n as u64,
            _ => span_bug!(e.span, "typeck error")
          };
          Repeat(Box::new(cx.eval(elem)?), n)
      },
      hir::ExprTupField(ref base, index) => {
        let c = cx.eval(base)?;
        if let Tuple(ref fields) = c {
            fields[index.node].clone()
        } else {
            signal!(base, ExpectedConstTuple);
        }
      }
      hir::ExprField(ref base, field_name) => {
        let c = cx.eval(base)?;
        if let Struct(ref fields) = c {
            if let Some(f) = fields.get(&field_name.node) {
                f.clone()
            } else {
                signal!(e, MissingStructField);
            }
        } else {
            signal!(base, ExpectedConstStruct);
        }
      }
      hir::ExprAddrOf(..) => signal!(e, UnimplementedConstVal("address operator")),
      _ => signal!(e, MiscCatchAll)
    };

    Ok(result)
}

fn resolve_trait_associated_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                            key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                                            -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let param_env = key.param_env;
    let (def_id, substs) = key.value;
    let trait_item = tcx.associated_item(def_id);
    let trait_id = trait_item.container.id();
    let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, substs));
    debug!("resolve_trait_associated_const: trait_ref={:?}",
           trait_ref);

    tcx.infer_ctxt().enter(|infcx| {
        let mut selcx = traits::SelectionContext::new(&infcx);
        let obligation = traits::Obligation::new(traits::ObligationCause::dummy(),
                                                 param_env,
                                                 trait_ref.to_poly_trait_predicate());
        let selection = match selcx.select(&obligation) {
            Ok(Some(vtable)) => vtable,
            // Still ambiguous, so give up and let the caller decide whether this
            // expression is really needed yet. Some associated constant values
            // can't be evaluated until monomorphization is done in trans.
            Ok(None) => {
                return None
            }
            Err(_) => {
                return None
            }
        };

        // NOTE: this code does not currently account for specialization, but when
        // it does so, it should hook into the param_env.reveal to determine when the
        // constant should resolve.
        match selection {
            traits::VtableImpl(ref impl_data) => {
                let name = trait_item.name;
                let ac = tcx.associated_items(impl_data.impl_def_id)
                    .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
                match ac {
                    // FIXME(eddyb) Use proper Instance resolution to
                    // get the correct Substs returned from here.
                    Some(ic) => {
                        let substs = Substs::identity_for_item(tcx, ic.def_id);
                        Some((ic.def_id, substs))
                    }
                    None => {
                        if trait_item.defaultness.has_value() {
                            Some(key.value)
                        } else {
                            None
                        }
                    }
                }
            }
            traits::VtableParam(_) => None,
            _ => {
                bug!("resolve_trait_associated_const: unexpected vtable type {:?}", selection)
            }
        }
    })
}

fn cast_const_int<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            val: ConstInt,
                            ty: Ty<'tcx>)
                            -> CastResult<'tcx> {
    let v = val.to_u128_unchecked();
    match ty.sty {
        ty::TyBool if v == 0 => Ok(Bool(false)),
        ty::TyBool if v == 1 => Ok(Bool(true)),
        ty::TyInt(ast::IntTy::I8) => Ok(Integral(I8(v as i128 as i8))),
        ty::TyInt(ast::IntTy::I16) => Ok(Integral(I16(v as i128 as i16))),
        ty::TyInt(ast::IntTy::I32) => Ok(Integral(I32(v as i128 as i32))),
        ty::TyInt(ast::IntTy::I64) => Ok(Integral(I64(v as i128 as i64))),
        ty::TyInt(ast::IntTy::I128) => Ok(Integral(I128(v as i128))),
        ty::TyInt(ast::IntTy::Is) => {
            Ok(Integral(Isize(ConstIsize::new_truncating(v as i128, tcx.sess.target.int_type))))
        },
        ty::TyUint(ast::UintTy::U8) => Ok(Integral(U8(v as u8))),
        ty::TyUint(ast::UintTy::U16) => Ok(Integral(U16(v as u16))),
        ty::TyUint(ast::UintTy::U32) => Ok(Integral(U32(v as u32))),
        ty::TyUint(ast::UintTy::U64) => Ok(Integral(U64(v as u64))),
        ty::TyUint(ast::UintTy::U128) => Ok(Integral(U128(v as u128))),
        ty::TyUint(ast::UintTy::Us) => {
            Ok(Integral(Usize(ConstUsize::new_truncating(v, tcx.sess.target.uint_type))))
        },
        ty::TyFloat(fty) => {
            if let Some(i) = val.to_u128() {
                Ok(Float(ConstFloat::from_u128(i, fty)))
            } else {
                // The value must be negative, go through signed integers.
                let i = val.to_u128_unchecked() as i128;
                Ok(Float(ConstFloat::from_i128(i, fty)))
            }
        }
        ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting an address to a raw ptr")),
        ty::TyChar => match val {
            U8(u) => Ok(Char(u as char)),
            _ => bug!(),
        },
        _ => Err(CannotCast),
    }
}

fn cast_const_float<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              val: ConstFloat,
                              ty: Ty<'tcx>) -> CastResult<'tcx> {
    let int_width = |ty| {
        ty::layout::Integer::from_attr(tcx, ty).size().bits() as usize
    };
    match ty.sty {
        ty::TyInt(ity) => {
            if let Some(i) = val.to_i128(int_width(attr::SignedInt(ity))) {
                cast_const_int(tcx, I128(i), ty)
            } else {
                Err(CannotCast)
            }
        }
        ty::TyUint(uty) => {
            if let Some(i) = val.to_u128(int_width(attr::UnsignedInt(uty))) {
                cast_const_int(tcx, U128(i), ty)
            } else {
                Err(CannotCast)
            }
        }
        ty::TyFloat(fty) => Ok(Float(val.convert(fty))),
        _ => Err(CannotCast),
    }
}

fn cast_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        val: ConstVal<'tcx>,
                        ty: Ty<'tcx>)
                        -> CastResult<'tcx> {
    match val {
        Integral(i) => cast_const_int(tcx, i, ty),
        Bool(b) => cast_const_int(tcx, U8(b as u8), ty),
        Float(f) => cast_const_float(tcx, f, ty),
        Char(c) => cast_const_int(tcx, U32(c as u32), ty),
        Variant(v) => {
            let adt = tcx.adt_def(tcx.parent_def_id(v).unwrap());
            let idx = adt.variant_index_with_id(v);
            cast_const_int(tcx, adt.discriminant_for_variant(tcx, idx), ty)
        }
        Function(..) => Err(UnimplementedConstVal("casting fn pointers")),
        ByteStr(b) => match ty.sty {
            ty::TyRawPtr(_) => {
                Err(ErrKind::UnimplementedConstVal("casting a bytestr to a raw ptr"))
            },
            ty::TyRef(_, ty::TypeAndMut { ref ty, mutbl: hir::MutImmutable }) => match ty.sty {
                ty::TyArray(ty, n) if ty == tcx.types.u8 && n == b.len() => Ok(ByteStr(b)),
                ty::TySlice(_) => {
                    Err(ErrKind::UnimplementedConstVal("casting a bytestr to slice"))
                },
                _ => Err(CannotCast),
            },
            _ => Err(CannotCast),
        },
        Str(s) => match ty.sty {
            ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting a str to a raw ptr")),
            ty::TyRef(_, ty::TypeAndMut { ref ty, mutbl: hir::MutImmutable }) => match ty.sty {
                ty::TyStr => Ok(Str(s)),
                _ => Err(CannotCast),
            },
            _ => Err(CannotCast),
        },
        _ => Err(CannotCast),
    }
}

fn lit_to_const<'a, 'tcx>(lit: &ast::LitKind,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          mut ty: Ty<'tcx>)
                          -> Result<ConstVal<'tcx>, ErrKind<'tcx>> {
    use syntax::ast::*;
    use syntax::ast::LitIntType::*;

    if let ty::TyAdt(adt, _) = ty.sty {
        if adt.is_enum() {
            ty = adt.repr.discr_type().to_ty(tcx)
        }
    }

    match *lit {
        LitKind::Str(ref s, _) => Ok(Str(s.as_str())),
        LitKind::ByteStr(ref data) => Ok(ByteStr(data.clone())),
        LitKind::Byte(n) => Ok(Integral(U8(n))),
        LitKind::Int(n, hint) => {
            match (&ty.sty, hint) {
                (&ty::TyInt(ity), _) |
                (_, Signed(ity)) => {
                    Ok(Integral(ConstInt::new_signed_truncating(n as i128,
                        ity, tcx.sess.target.int_type)))
                }
                (&ty::TyUint(uty), _) |
                (_, Unsigned(uty)) => {
                    Ok(Integral(ConstInt::new_unsigned_truncating(n as u128,
                        uty, tcx.sess.target.uint_type)))
                }
                _ => bug!()
            }
        }
        LitKind::Float(n, fty) => {
            parse_float(&n.as_str(), fty).map(Float)
        }
        LitKind::FloatUnsuffixed(n) => {
            let fty = match ty.sty {
                ty::TyFloat(fty) => fty,
                _ => bug!()
            };
            parse_float(&n.as_str(), fty).map(Float)
        }
        LitKind::Bool(b) => Ok(Bool(b)),
        LitKind::Char(c) => Ok(Char(c)),
    }
}

fn parse_float<'tcx>(num: &str, fty: ast::FloatTy)
                     -> Result<ConstFloat, ErrKind<'tcx>> {
    ConstFloat::from_str(num, fty).map_err(|_| {
        // FIXME(#31407) this is only necessary because float parsing is buggy
        UnimplementedConstVal("could not evaluate float literal (see issue #31407)")
    })
}

pub fn compare_const_vals(tcx: TyCtxt, span: Span, a: &ConstVal, b: &ConstVal)
                          -> Result<Ordering, ErrorReported>
{
    let result = match (a, b) {
        (&Integral(a), &Integral(b)) => a.try_cmp(b).ok(),
        (&Float(a), &Float(b)) => a.try_cmp(b).ok(),
        (&Str(ref a), &Str(ref b)) => Some(a.cmp(b)),
        (&Bool(a), &Bool(b)) => Some(a.cmp(&b)),
        (&ByteStr(ref a), &ByteStr(ref b)) => Some(a.cmp(b)),
        (&Char(a), &Char(ref b)) => Some(a.cmp(b)),
        _ => None,
    };

    match result {
        Some(result) => Ok(result),
        None => {
            // FIXME: can this ever be reached?
            span_err!(tcx.sess, span, E0298,
                      "type mismatch comparing {} and {}",
                      a.description(),
                      b.description());
            Err(ErrorReported)
        }
    }
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn compare_lit_exprs(&self,
                             span: Span,
                             a: &Expr,
                             b: &Expr) -> Result<Ordering, ErrorReported> {
        let tcx = self.tcx;
        let a = match self.eval(a) {
            Ok(a) => a,
            Err(e) => {
                e.report(tcx, a.span, "expression");
                return Err(ErrorReported);
            }
        };
        let b = match self.eval(b) {
            Ok(b) => b,
            Err(e) => {
                e.report(tcx, b.span, "expression");
                return Err(ErrorReported);
            }
        };
        compare_const_vals(tcx, span, &a, &b)
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        const_eval,
        ..*providers
    };
}

fn const_eval<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                        -> EvalResult<'tcx> {
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
        tcx.mir_const_qualif(def_id);
        tcx.hir.body(tcx.hir.body_owned_by(id))
    } else {
        tcx.sess.cstore.item_body(tcx, def_id)
    };
    ConstContext::new(tcx, key.param_env.and(substs), tables).eval(&body.value)
}
