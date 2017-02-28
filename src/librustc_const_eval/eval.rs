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
use rustc::middle::const_val::ConstVal;
use self::ErrKind::*;

use rustc::hir::map as hir_map;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::util::IntTypeExt;
use rustc::ty::subst::{Substs, Subst};
use rustc::traits::Reveal;
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::DefIdMap;

use graphviz::IntoCow;
use syntax::ast;
use rustc::hir::{self, Expr};
use syntax_pos::{Span, DUMMY_SP};

use std::borrow::Cow;
use std::cmp::Ordering;

use rustc_const_math::*;
use rustc_errors::DiagnosticBuilder;

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

fn lookup_variant_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  variant_def: DefId)
                                  -> Option<(&'tcx Expr, &'a ty::TypeckTables<'tcx>)> {
    if let Some(variant_node_id) = tcx.hir.as_local_node_id(variant_def) {
        let enum_node_id = tcx.hir.get_parent(variant_node_id);
        if let Some(hir_map::NodeItem(it)) = tcx.hir.find(enum_node_id) {
            if let hir::ItemEnum(ref edef, _) = it.node {
                for variant in &edef.variants {
                    if variant.node.data.id() == variant_node_id {
                        return variant.node.disr_expr.map(|e| {
                            let def_id = tcx.hir.body_owner_def_id(e);
                            (&tcx.hir.body(e).value,
                             tcx.item_tables(def_id))
                        });
                    }
                }
            }
        }
    }
    None
}

/// * `def_id` is the id of the constant.
/// * `substs` is the monomorphized substitutions for the expression.
///
/// `substs` is optional and is used for associated constants.
/// This generally happens in late/trans const evaluation.
pub fn lookup_const_by_id<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        def_id: DefId,
                                        substs: &'tcx Substs<'tcx>)
                                        -> Option<(&'tcx Expr,
                                                   &'a ty::TypeckTables<'tcx>)> {
    if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
        match tcx.hir.find(node_id) {
            None => None,
            Some(hir_map::NodeItem(&hir::Item {
                node: hir::ItemConst(_, body), ..
            })) |
            Some(hir_map::NodeImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Const(_, body), ..
            })) => {
                Some((&tcx.hir.body(body).value,
                      tcx.item_tables(def_id)))
            }
            Some(hir_map::NodeTraitItem(ti)) => match ti.node {
                hir::TraitItemKind::Const(_, default) => {
                    // If we have a trait item and the substitutions for it,
                    // `resolve_trait_associated_const` will select an impl
                    // or the default.
                    let trait_id = tcx.hir.get_parent(node_id);
                    let trait_id = tcx.hir.local_def_id(trait_id);
                    let default_value = default.map(|body| {
                        (&tcx.hir.body(body).value,
                            tcx.item_tables(def_id))
                    });
                    resolve_trait_associated_const(tcx, def_id, default_value, trait_id, substs)
                }
                _ => None
            },
            Some(_) => None
        }
    } else {
        let expr_and_tables = tcx.sess.cstore.maybe_get_item_body(tcx, def_id).map(|body| {
            (&body.value, tcx.item_tables(def_id))
        });
        match tcx.sess.cstore.describe_def(def_id) {
            Some(Def::AssociatedConst(_)) => {
                let trait_id = tcx.sess.cstore.trait_of_item(def_id);
                // As mentioned in the comments above for in-crate
                // constants, we only try to find the expression for a
                // trait-associated const if the caller gives us the
                // substitutions for the reference to it.
                if let Some(trait_id) = trait_id {
                    resolve_trait_associated_const(tcx, def_id, expr_and_tables,
                                                   trait_id, substs)
                } else {
                    expr_and_tables
                }
            },
            Some(Def::Const(..)) => expr_and_tables,
            _ => None
        }
    }
}

fn lookup_const_fn_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                                   -> Option<(&'tcx hir::Body, &'a ty::TypeckTables<'tcx>)>
{
    if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
        FnLikeNode::from_node(tcx.hir.get(node_id)).and_then(|fn_like| {
            if fn_like.constness() == hir::Constness::Const {
                Some((tcx.hir.body(fn_like.body()),
                      tcx.item_tables(def_id)))
            } else {
                None
            }
        })
    } else {
        if tcx.sess.cstore.is_const_fn(def_id) {
            tcx.sess.cstore.maybe_get_item_body(tcx, def_id).map(|body| {
                (body, tcx.item_tables(def_id))
            })
        } else {
            None
        }
    }
}

fn build_const_eval_err<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    err: &ConstEvalErr,
    primary_span: Span,
    primary_kind: &str)
    -> DiagnosticBuilder<'tcx>
{
    let mut err = err;
    while let &ConstEvalErr { kind: ErroneousReferencedConstant(box ref i_err), .. } = err {
        err = i_err;
    }

    let mut diag = struct_span_err!(tcx.sess, err.span, E0080, "constant evaluation error");
    note_const_eval_err(tcx, err, primary_span, primary_kind, &mut diag);
    diag
}

pub fn report_const_eval_err<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    err: &ConstEvalErr,
    primary_span: Span,
    primary_kind: &str)
{
    if let TypeckError = err.kind {
        return;
    }
    build_const_eval_err(tcx, err, primary_span, primary_kind).emit();
}

pub fn fatal_const_eval_err<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    err: &ConstEvalErr,
    primary_span: Span,
    primary_kind: &str)
    -> !
{
    report_const_eval_err(tcx, err, primary_span, primary_kind);
    tcx.sess.abort_if_errors();
    unreachable!()
}

pub fn note_const_eval_err<'a, 'tcx>(
    _tcx: TyCtxt<'a, 'tcx, 'tcx>,
    err: &ConstEvalErr,
    primary_span: Span,
    primary_kind: &str,
    diag: &mut DiagnosticBuilder)
{
    match err.description() {
        ConstEvalErrDescription::Simple(message) => {
            diag.span_label(err.span, &message);
        }
    }

    if !primary_span.contains(err.span) {
        diag.span_note(primary_span,
                       &format!("for {} here", primary_kind));
    }
}

pub struct ConstContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    substs: &'tcx Substs<'tcx>,
    fn_args: Option<DefIdMap<ConstVal<'tcx>>>
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, body: hir::BodyId) -> Self {
        let def_id = tcx.hir.body_owner_def_id(body);
        ty::queries::mir_const_qualif::get(tcx, DUMMY_SP, def_id);
        ConstContext::with_tables(tcx, tcx.item_tables(def_id))
    }

    pub fn with_tables(tcx: TyCtxt<'a, 'tcx, 'tcx>, tables: &'a ty::TypeckTables<'tcx>) -> Self {
        ConstContext {
            tcx: tcx,
            tables: tables,
            substs: tcx.intern_substs(&[]),
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

#[derive(Clone, Debug)]
pub struct ConstEvalErr<'tcx> {
    pub span: Span,
    pub kind: ErrKind<'tcx>,
}

#[derive(Clone, Debug)]
pub enum ErrKind<'tcx> {
    CannotCast,
    MissingStructField,
    NegateOn(ConstVal<'tcx>),
    NotOn(ConstVal<'tcx>),
    CallOn(ConstVal<'tcx>),

    NonConstPath,
    UnimplementedConstVal(&'static str),
    ExpectedConstTuple,
    ExpectedConstStruct,
    IndexedNonVec,
    IndexNotUsize,
    IndexOutOfBounds { len: u64, index: u64 },

    MiscBinaryOp,
    MiscCatchAll,

    IndexOpFeatureGated,
    Math(ConstMathErr),

    ErroneousReferencedConstant(Box<ConstEvalErr<'tcx>>),

    TypeckError
}

impl<'tcx> From<ConstMathErr> for ErrKind<'tcx> {
    fn from(err: ConstMathErr) -> ErrKind<'tcx> {
        match err {
            ConstMathErr::UnsignedNegation => TypeckError,
            _ => Math(err)
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConstEvalErrDescription<'a> {
    Simple(Cow<'a, str>),
}

impl<'a> ConstEvalErrDescription<'a> {
    /// Return a one-line description of the error, for lints and such
    pub fn into_oneline(self) -> Cow<'a, str> {
        match self {
            ConstEvalErrDescription::Simple(simple) => simple,
        }
    }
}

impl<'tcx> ConstEvalErr<'tcx> {
    pub fn description(&self) -> ConstEvalErrDescription {
        use self::ErrKind::*;
        use self::ConstEvalErrDescription::*;

        macro_rules! simple {
            ($msg:expr) => ({ Simple($msg.into_cow()) });
            ($fmt:expr, $($arg:tt)+) => ({
                Simple(format!($fmt, $($arg)+).into_cow())
            })
        }

        match self.kind {
            CannotCast => simple!("can't cast this type"),
            NegateOn(ref const_val) => simple!("negate on {}", const_val.description()),
            NotOn(ref const_val) => simple!("not on {}", const_val.description()),
            CallOn(ref const_val) => simple!("call on {}", const_val.description()),

            MissingStructField  => simple!("nonexistent struct field"),
            NonConstPath        => simple!("non-constant path in constant expression"),
            UnimplementedConstVal(what) =>
                simple!("unimplemented constant expression: {}", what),
            ExpectedConstTuple => simple!("expected constant tuple"),
            ExpectedConstStruct => simple!("expected constant struct"),
            IndexedNonVec => simple!("indexing is only supported for arrays"),
            IndexNotUsize => simple!("indices must be of type `usize`"),
            IndexOutOfBounds { len, index } => {
                simple!("index out of bounds: the len is {} but the index is {}",
                        len, index)
            }

            MiscBinaryOp => simple!("bad operands for binary"),
            MiscCatchAll => simple!("unsupported constant expr"),
            IndexOpFeatureGated => simple!("the index operation on const values is unstable"),
            Math(ref err) => Simple(err.description().into_cow()),

            ErroneousReferencedConstant(_) => simple!("could not evaluate referenced constant"),

            TypeckError => simple!("type-checking failed"),
        }
    }
}

pub type EvalResult<'tcx> = Result<ConstVal<'tcx>, ConstEvalErr<'tcx>>;
pub type CastResult<'tcx> = Result<ConstVal<'tcx>, ErrKind<'tcx>>;

fn eval_const_expr_partial<'a, 'tcx>(cx: &ConstContext<'a, 'tcx>,
                                     e: &Expr) -> EvalResult<'tcx> {
    let tcx = cx.tcx;
    let ety = cx.tables.expr_ty(e);

    // Avoid applying substitutions if they're empty, that'd ICE.
    let ety = if cx.substs.is_empty() {
        ety
    } else {
        ety.subst(tcx, cx.substs)
    };

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

          _ => signal!(e, MiscBinaryOp),
        }
      }
      hir::ExprCast(ref base, _) => {
        match cast_const(tcx, cx.eval(base)?, ety) {
            Ok(val) => val,
            Err(kind) => return Err(ConstEvalErr { span: e.span, kind: kind }),
        }
      }
      hir::ExprPath(ref qpath) => {
        let substs = cx.tables.node_id_item_substs(e.id)
            .unwrap_or_else(|| tcx.intern_substs(&[]));

        // Avoid applying substitutions if they're empty, that'd ICE.
        let substs = if cx.substs.is_empty() {
            substs
        } else {
            substs.subst(tcx, cx.substs)
        };

          match cx.tables.qpath_def(qpath, e.id) {
              Def::Const(def_id) |
              Def::AssociatedConst(def_id) => {
                  if let Some((expr, tables)) = lookup_const_by_id(tcx, def_id, substs) {
                      let cx = ConstContext::with_tables(tcx, tables);
                      match cx.eval(expr) {
                          Ok(val) => val,
                          Err(ConstEvalErr { kind: TypeckError, .. }) => {
                              signal!(e, TypeckError);
                          }
                          Err(err) => {
                              debug!("bad reference: {:?}, {:?}", err.description(), err.span);
                              signal!(e, ErroneousReferencedConstant(box err))
                          },
                      }
                  } else {
                      signal!(e, TypeckError);
                  }
              },
              Def::VariantCtor(variant_def, ..) => {
                  if let Some((expr, tables)) = lookup_variant_by_id(tcx, variant_def) {
                      let cx = ConstContext::with_tables(tcx, tables);
                      match cx.eval(expr) {
                          Ok(val) => val,
                          Err(ConstEvalErr { kind: TypeckError, .. }) => {
                              signal!(e, TypeckError);
                          }
                          Err(err) => {
                              debug!("bad reference: {:?}, {:?}", err.description(), err.span);
                              signal!(e, ErroneousReferencedConstant(box err))
                          },
                      }
                  } else {
                      signal!(e, UnimplementedConstVal("enum variants"));
                  }
              }
              Def::StructCtor(..) => {
                  ConstVal::Struct(Default::default())
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
          let (did, substs) = match cx.eval(callee)? {
              Function(did, substs) => (did, substs),
              Struct(_) => signal!(e, UnimplementedConstVal("tuple struct constructors")),
              callee => signal!(e, CallOn(callee)),
          };
          let (body, tables) = match lookup_const_fn_by_id(tcx, did) {
              Some(x) => x,
              None => signal!(e, NonConstPath),
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
            tcx: tcx,
            tables: tables,
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

fn resolve_trait_associated_const<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    trait_item_id: DefId,
    default_value: Option<(&'tcx Expr, &'a ty::TypeckTables<'tcx>)>,
    trait_id: DefId,
    rcvr_substs: &'tcx Substs<'tcx>
) -> Option<(&'tcx Expr, &'a ty::TypeckTables<'tcx>)>
{
    let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, rcvr_substs));
    debug!("resolve_trait_associated_const: trait_ref={:?}",
           trait_ref);

    tcx.populate_implementations_for_trait_if_necessary(trait_id);
    tcx.infer_ctxt((), Reveal::UserFacing).enter(|infcx| {
        let mut selcx = traits::SelectionContext::new(&infcx);
        let obligation = traits::Obligation::new(traits::ObligationCause::dummy(),
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
        // it does so, it should hook into the Reveal to determine when the
        // constant should resolve; this will also require plumbing through to this
        // function whether we are in "trans mode" to pick the right Reveal
        // when constructing the inference context above.
        match selection {
            traits::VtableImpl(ref impl_data) => {
                let name = tcx.associated_item(trait_item_id).name;
                let ac = tcx.associated_items(impl_data.impl_def_id)
                    .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
                match ac {
                    Some(ic) => lookup_const_by_id(tcx, ic.def_id, Substs::empty()),
                    None => default_value,
                }
            }
            _ => {
                bug!("resolve_trait_associated_const: unexpected vtable type")
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
        ty::TyFloat(ast::FloatTy::F64) => Ok(Float(F64(val.to_f64()))),
        ty::TyFloat(ast::FloatTy::F32) => Ok(Float(F32(val.to_f32()))),
        ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting an address to a raw ptr")),
        ty::TyChar => match val {
            U8(u) => Ok(Char(u as char)),
            _ => bug!(),
        },
        _ => bug!(),
    }
}

fn cast_const_float<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              val: ConstFloat,
                              ty: Ty<'tcx>) -> CastResult<'tcx> {
    match ty.sty {
        ty::TyInt(_) | ty::TyUint(_) => {
            let i = match val {
                F32(f) if f >= 0.0 => U128(f as u128),
                F64(f) if f >= 0.0 => U128(f as u128),

                F32(f) => I128(f as i128),
                F64(f) => I128(f as i128)
            };

            if let (I128(_), &ty::TyUint(_)) = (i, &ty.sty) {
                return Err(CannotCast);
            }

            cast_const_int(tcx, i, ty)
        }
        ty::TyFloat(ast::FloatTy::F64) => Ok(Float(F64(match val {
            F32(f) => f as f64,
            F64(f) => f
        }))),
        ty::TyFloat(ast::FloatTy::F32) => Ok(Float(F32(match val {
            F64(f) => f as f32,
            F32(f) => f
        }))),
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
    let val = match fty {
        ast::FloatTy::F32 => num.parse::<f32>().map(F32),
        ast::FloatTy::F64 => num.parse::<f64>().map(F64)
    };
    val.map_err(|_| {
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
                report_const_eval_err(tcx, &e, a.span, "expression");
                return Err(ErrorReported);
            }
        };
        let b = match self.eval(b) {
            Ok(b) => b,
            Err(e) => {
                report_const_eval_err(tcx, &e, b.span, "expression");
                return Err(ErrorReported);
            }
        };
        compare_const_vals(tcx, span, &a, &b)
    }
}


/// Returns the value of the length-valued expression
pub fn eval_length<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             count: hir::BodyId,
                             reason: &str)
                             -> Result<usize, ErrorReported>
{
    let count_expr = &tcx.hir.body(count).value;
    match ConstContext::new(tcx, count).eval(count_expr) {
        Ok(Integral(Usize(count))) => {
            let val = count.as_u64(tcx.sess.target.uint_type);
            assert_eq!(val as usize as u64, val);
            Ok(val as usize)
        },
        Ok(_) |
        Err(ConstEvalErr { kind: TypeckError, .. }) => Err(ErrorReported),
        Err(err) => {
            let mut diag = build_const_eval_err(
                tcx, &err, count_expr.span, reason);

            if let hir::ExprPath(hir::QPath::Resolved(None, ref path)) = count_expr.node {
                if let Def::Local(..) = path.def {
                    diag.note(&format!("`{}` is a variable",
                                       tcx.hir.node_to_pretty_string(count_expr.id)));
                }
            }

            diag.emit();
            Err(ErrorReported)
        }
    }
}
