// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//#![allow(non_camel_case_types)]

use rustc::middle::const_val::ConstVal::*;
use rustc::middle::const_val::ConstVal;
use self::ErrKind::*;
use self::EvalHint::*;

use rustc::hir::map as ast_map;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::util::IntTypeExt;
use rustc::ty::subst::Substs;
use rustc::traits::Reveal;
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::DefIdMap;

use graphviz::IntoCow;
use syntax::ast;
use rustc::hir::{self, Expr};
use syntax::attr::IntType;
use syntax_pos::Span;

use std::borrow::Cow;
use std::cmp::Ordering;

use rustc_const_math::*;
use rustc_errors::DiagnosticBuilder;

use rustc_i128::{i128, u128};

macro_rules! math {
    ($e:expr, $op:expr) => {
        match $op {
            Ok(val) => val,
            Err(e) => signal!($e, Math(e)),
        }
    }
}

fn lookup_variant_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  variant_def: DefId)
                                  -> Option<(&'tcx Expr, Option<&'a ty::Tables<'tcx>>)> {
    if let Some(variant_node_id) = tcx.map.as_local_node_id(variant_def) {
        let enum_node_id = tcx.map.get_parent(variant_node_id);
        if let Some(ast_map::NodeItem(it)) = tcx.map.find(enum_node_id) {
            if let hir::ItemEnum(ref edef, _) = it.node {
                for variant in &edef.variants {
                    if variant.node.data.id() == variant_node_id {
                        return variant.node.disr_expr.map(|e| {
                            let def_id = tcx.map.body_owner_def_id(e);
                            (&tcx.map.body(e).value,
                             tcx.tables.borrow().get(&def_id).cloned())
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
                                        substs: Option<&'tcx Substs<'tcx>>)
                                        -> Option<(&'tcx Expr,
                                                   Option<&'a ty::Tables<'tcx>>,
                                                   Option<ty::Ty<'tcx>>)> {
    if let Some(node_id) = tcx.map.as_local_node_id(def_id) {
        match tcx.map.find(node_id) {
            None => None,
            Some(ast_map::NodeItem(&hir::Item {
                node: hir::ItemConst(ref ty, body), ..
            })) |
            Some(ast_map::NodeImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Const(ref ty, body), ..
            })) => {
                Some((&tcx.map.body(body).value,
                      tcx.tables.borrow().get(&def_id).cloned(),
                      tcx.ast_ty_to_prim_ty(ty)))
            }
            Some(ast_map::NodeTraitItem(ti)) => match ti.node {
                hir::TraitItemKind::Const(ref ty, default) => {
                    if let Some(substs) = substs {
                        // If we have a trait item and the substitutions for it,
                        // `resolve_trait_associated_const` will select an impl
                        // or the default.
                        let trait_id = tcx.map.get_parent(node_id);
                        let trait_id = tcx.map.local_def_id(trait_id);
                        let default_value = default.map(|body| {
                            (&tcx.map.body(body).value,
                             tcx.tables.borrow().get(&def_id).cloned(),
                             tcx.ast_ty_to_prim_ty(ty))
                        });
                        resolve_trait_associated_const(tcx, def_id, default_value, trait_id, substs)
                    } else {
                        // Technically, without knowing anything about the
                        // expression that generates the obligation, we could
                        // still return the default if there is one. However,
                        // it's safer to return `None` than to return some value
                        // that may differ from what you would get from
                        // correctly selecting an impl.
                        None
                    }
                }
                _ => None
            },
            Some(_) => None
        }
    } else {
        let expr_tables_ty = tcx.sess.cstore.maybe_get_item_body(tcx, def_id).map(|body| {
            (&body.value, Some(tcx.item_tables(def_id)),
             Some(tcx.sess.cstore.item_type(tcx, def_id)))
        });
        match tcx.sess.cstore.describe_def(def_id) {
            Some(Def::AssociatedConst(_)) => {
                let trait_id = tcx.sess.cstore.trait_of_item(def_id);
                // As mentioned in the comments above for in-crate
                // constants, we only try to find the expression for a
                // trait-associated const if the caller gives us the
                // substitutions for the reference to it.
                if let Some(trait_id) = trait_id {
                    if let Some(substs) = substs {
                        resolve_trait_associated_const(tcx, def_id, expr_tables_ty,
                                                       trait_id, substs)
                    } else {
                        None
                    }
                } else {
                    expr_tables_ty
                }
            },
            Some(Def::Const(..)) => expr_tables_ty,
            _ => None
        }
    }
}

fn lookup_const_fn_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                                   -> Option<(&'tcx hir::Body, Option<&'a ty::Tables<'tcx>>)>
{
    if let Some(node_id) = tcx.map.as_local_node_id(def_id) {
        FnLikeNode::from_node(tcx.map.get(node_id)).and_then(|fn_like| {
            if fn_like.constness() == hir::Constness::Const {
                Some((tcx.map.body(fn_like.body()),
                      tcx.tables.borrow().get(&def_id).cloned()))
            } else {
                None
            }
        })
    } else {
        if tcx.sess.cstore.is_const_fn(def_id) {
            tcx.sess.cstore.maybe_get_item_body(tcx, def_id).map(|body| {
                (body, Some(tcx.item_tables(def_id)))
            })
        } else {
            None
        }
    }
}

pub fn report_const_eval_err<'a, 'tcx>(
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

pub fn fatal_const_eval_err<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    err: &ConstEvalErr,
    primary_span: Span,
    primary_kind: &str)
    -> !
{
    report_const_eval_err(tcx, err, primary_span, primary_kind).emit();
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
    tables: Option<&'a ty::Tables<'tcx>>,
    fn_args: Option<DefIdMap<ConstVal>>
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, body: hir::BodyId) -> Self {
        let def_id = tcx.map.body_owner_def_id(body);
        ConstContext {
            tcx: tcx,
            tables: tcx.tables.borrow().get(&def_id).cloned(),
            fn_args: None
        }
    }

    pub fn with_tables(tcx: TyCtxt<'a, 'tcx, 'tcx>, tables: &'a ty::Tables<'tcx>) -> Self {
        ConstContext {
            tcx: tcx,
            tables: Some(tables),
            fn_args: None
        }
    }

    /// Evaluate a constant expression in a context where the expression isn't
    /// guaranteed to be evaluatable. `ty_hint` is usually ExprTypeChecked,
    /// but a few places need to evaluate constants during type-checking, like
    /// computing the length of an array. (See also the FIXME above EvalHint.)
    pub fn eval(&self, e: &Expr, ty_hint: EvalHint<'tcx>) -> EvalResult {
        eval_const_expr_partial(self, e, ty_hint)
    }
}

#[derive(Clone, Debug)]
pub struct ConstEvalErr {
    pub span: Span,
    pub kind: ErrKind,
}

#[derive(Clone, Debug)]
pub enum ErrKind {
    CannotCast,
    CannotCastTo(&'static str),
    InvalidOpForInts(hir::BinOp_),
    InvalidOpForBools(hir::BinOp_),
    InvalidOpForFloats(hir::BinOp_),
    InvalidOpForIntUint(hir::BinOp_),
    InvalidOpForUintInt(hir::BinOp_),
    NegateOn(ConstVal),
    NotOn(ConstVal),
    CallOn(ConstVal),

    MissingStructField,
    NonConstPath,
    UnimplementedConstVal(&'static str),
    UnresolvedPath,
    ExpectedConstTuple,
    ExpectedConstStruct,
    TupleIndexOutOfBounds,
    IndexedNonVec,
    IndexNegative,
    IndexNotInt,
    IndexOutOfBounds { len: u64, index: u64 },
    RepeatCountNotNatural,
    RepeatCountNotInt,

    MiscBinaryOp,
    MiscCatchAll,

    IndexOpFeatureGated,
    Math(ConstMathErr),

    IntermediateUnsignedNegative,
    /// Expected, Got
    TypeMismatch(String, ConstInt),

    BadType(ConstVal),
    ErroneousReferencedConstant(Box<ConstEvalErr>),
    CharCast(ConstInt),
}

impl From<ConstMathErr> for ErrKind {
    fn from(err: ConstMathErr) -> ErrKind {
        Math(err)
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

impl ConstEvalErr {
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
            CannotCastTo(s) => simple!("can't cast this type to {}", s),
            InvalidOpForInts(_) =>  simple!("can't do this op on integrals"),
            InvalidOpForBools(_) =>  simple!("can't do this op on bools"),
            InvalidOpForFloats(_) => simple!("can't do this op on floats"),
            InvalidOpForIntUint(..) => simple!("can't do this op on an isize and usize"),
            InvalidOpForUintInt(..) => simple!("can't do this op on a usize and isize"),
            NegateOn(ref const_val) => simple!("negate on {}", const_val.description()),
            NotOn(ref const_val) => simple!("not on {}", const_val.description()),
            CallOn(ref const_val) => simple!("call on {}", const_val.description()),

            MissingStructField  => simple!("nonexistent struct field"),
            NonConstPath        => simple!("non-constant path in constant expression"),
            UnimplementedConstVal(what) =>
                simple!("unimplemented constant expression: {}", what),
            UnresolvedPath => simple!("unresolved path in constant expression"),
            ExpectedConstTuple => simple!("expected constant tuple"),
            ExpectedConstStruct => simple!("expected constant struct"),
            TupleIndexOutOfBounds => simple!("tuple index out of bounds"),
            IndexedNonVec => simple!("indexing is only supported for arrays"),
            IndexNegative => simple!("indices must be non-negative integers"),
            IndexNotInt => simple!("indices must be integers"),
            IndexOutOfBounds { len, index } => {
                simple!("index out of bounds: the len is {} but the index is {}",
                        len, index)
            }
            RepeatCountNotNatural => simple!("repeat count must be a natural number"),
            RepeatCountNotInt => simple!("repeat count must be integers"),

            MiscBinaryOp => simple!("bad operands for binary"),
            MiscCatchAll => simple!("unsupported constant expr"),
            IndexOpFeatureGated => simple!("the index operation on const values is unstable"),
            Math(ref err) => Simple(err.description().into_cow()),

            IntermediateUnsignedNegative => simple!(
                "during the computation of an unsigned a negative \
                 number was encountered. This is most likely a bug in\
                 the constant evaluator"),

            TypeMismatch(ref expected, ref got) => {
                simple!("expected {}, found {}", expected, got.description())
            },
            BadType(ref i) => simple!("value of wrong type: {:?}", i),
            ErroneousReferencedConstant(_) => simple!("could not evaluate referenced constant"),
            CharCast(ref got) => {
                simple!("only `u8` can be cast as `char`, not `{}`", got.description())
            },
        }
    }
}

pub type EvalResult = Result<ConstVal, ConstEvalErr>;
pub type CastResult = Result<ConstVal, ErrKind>;

// FIXME: Long-term, this enum should go away: trying to evaluate
// an expression which hasn't been type-checked is a recipe for
// disaster.  That said, it's not clear how to fix ast_ty_to_ty
// to avoid the ordering issue.

/// Hint to determine how to evaluate constant expressions which
/// might not be type-checked.
#[derive(Copy, Clone, Debug)]
pub enum EvalHint<'tcx> {
    /// We have a type-checked expression.
    ExprTypeChecked,
    /// We have an expression which hasn't been type-checked, but we have
    /// an idea of what the type will be because of the context. For example,
    /// the length of an array is always `usize`. (This is referred to as
    /// a hint because it isn't guaranteed to be consistent with what
    /// type-checking would compute.)
    UncheckedExprHint(Ty<'tcx>),
    /// We have an expression which has not yet been type-checked, and
    /// and we have no clue what the type will be.
    UncheckedExprNoHint,
}

impl<'tcx> EvalHint<'tcx> {
    fn erase_hint(&self) -> EvalHint<'tcx> {
        match *self {
            ExprTypeChecked => ExprTypeChecked,
            UncheckedExprHint(_) | UncheckedExprNoHint => UncheckedExprNoHint,
        }
    }
    fn checked_or(&self, ty: Ty<'tcx>) -> EvalHint<'tcx> {
        match *self {
            ExprTypeChecked => ExprTypeChecked,
            _ => UncheckedExprHint(ty),
        }
    }
}

macro_rules! signal {
    ($e:expr, $exn:expr) => {
        return Err(ConstEvalErr { span: $e.span, kind: $exn })
    }
}

fn eval_const_expr_partial<'a, 'tcx>(cx: &ConstContext<'a, 'tcx>,
                                     e: &Expr,
                                     ty_hint: EvalHint<'tcx>) -> EvalResult {
    let tcx = cx.tcx;
    // Try to compute the type of the expression based on the EvalHint.
    // (See also the definition of EvalHint, and the FIXME above EvalHint.)
    let ety = match ty_hint {
        ExprTypeChecked => {
            // After type-checking, expr_ty is guaranteed to succeed.
            cx.tables.map(|tables| tables.expr_ty(e))
        }
        UncheckedExprHint(ty) => {
            // Use the type hint; it's not guaranteed to be right, but it's
            // usually good enough.
            Some(ty)
        }
        UncheckedExprNoHint => {
            // This expression might not be type-checked, and we have no hint.
            // Try to query the context for a type anyway; we might get lucky
            // (for example, if the expression was imported from another crate).
            cx.tables.and_then(|tables| tables.expr_ty_opt(e))
        }
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
            match (&lit.node, ety.map(|t| &t.sty)) {
                (&LitKind::Int(I8_OVERFLOW, _), Some(&ty::TyInt(IntTy::I8))) |
                (&LitKind::Int(I8_OVERFLOW, Signed(IntTy::I8)), _) => {
                    return Ok(Integral(I8(i8::min_value())))
                },
                (&LitKind::Int(I16_OVERFLOW, _), Some(&ty::TyInt(IntTy::I16))) |
                (&LitKind::Int(I16_OVERFLOW, Signed(IntTy::I16)), _) => {
                    return Ok(Integral(I16(i16::min_value())))
                },
                (&LitKind::Int(I32_OVERFLOW, _), Some(&ty::TyInt(IntTy::I32))) |
                (&LitKind::Int(I32_OVERFLOW, Signed(IntTy::I32)), _) => {
                    return Ok(Integral(I32(i32::min_value())))
                },
                (&LitKind::Int(I64_OVERFLOW, _), Some(&ty::TyInt(IntTy::I64))) |
                (&LitKind::Int(I64_OVERFLOW, Signed(IntTy::I64)), _) => {
                    return Ok(Integral(I64(i64::min_value())))
                },
                (&LitKind::Int(n, _), Some(&ty::TyInt(IntTy::I128))) |
                (&LitKind::Int(n, Signed(IntTy::I128)), _) => {
                    // SNAP: replace n in pattern with I128_OVERFLOW and remove this if.
                    if n == I128_OVERFLOW {
                        return Ok(Integral(I128(i128::min_value())))
                    }
                },
                (&LitKind::Int(n, _), Some(&ty::TyInt(IntTy::Is))) |
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
                        _ => bug!(),
                    }
                },
                _ => {},
            }
        }
        match cx.eval(inner, ty_hint)? {
          Float(f) => Float(-f),
          Integral(i) => Integral(math!(e, -i)),
          const_val => signal!(e, NegateOn(const_val)),
        }
      }
      hir::ExprUnary(hir::UnNot, ref inner) => {
        match cx.eval(inner, ty_hint)? {
          Integral(i) => Integral(math!(e, !i)),
          Bool(b) => Bool(!b),
          const_val => signal!(e, NotOn(const_val)),
        }
      }
      hir::ExprUnary(hir::UnDeref, _) => signal!(e, UnimplementedConstVal("deref operation")),
      hir::ExprBinary(op, ref a, ref b) => {
        let b_ty = match op.node {
            hir::BiShl | hir::BiShr => ty_hint.erase_hint(),
            _ => ty_hint
        };
        // technically, if we don't have type hints, but integral eval
        // gives us a type through a type-suffix, cast or const def type
        // we need to re-eval the other value of the BinOp if it was
        // not inferred
        match (cx.eval(a, ty_hint)?,
               cx.eval(b, b_ty)?) {
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
              _ => signal!(e, InvalidOpForFloats(op.node)),
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
              _ => signal!(e, InvalidOpForInts(op.node)),
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
              _ => signal!(e, InvalidOpForBools(op.node)),
             })
          }

          _ => signal!(e, MiscBinaryOp),
        }
      }
      hir::ExprCast(ref base, ref target_ty) => {
        let ety = tcx.ast_ty_to_prim_ty(&target_ty).or(ety)
                .unwrap_or_else(|| {
                    tcx.sess.span_fatal(target_ty.span,
                                        "target type not found for const cast")
                });

        let base_hint = if let ExprTypeChecked = ty_hint {
            ExprTypeChecked
        } else {
            match cx.tables.and_then(|tables| tables.expr_ty_opt(&base)) {
                Some(t) => UncheckedExprHint(t),
                None => ty_hint
            }
        };

        let val = match cx.eval(base, base_hint) {
            Ok(val) => val,
            Err(ConstEvalErr { kind: ErroneousReferencedConstant(
                box ConstEvalErr { kind: TypeMismatch(_, val), .. }), .. }) |
            Err(ConstEvalErr { kind: TypeMismatch(_, val), .. }) => {
                // Something like `5i8 as usize` doesn't need a type hint for the base
                // instead take the type hint from the inner value
                let hint = match val.int_type() {
                    Some(IntType::UnsignedInt(ty)) => ty_hint.checked_or(tcx.mk_mach_uint(ty)),
                    Some(IntType::SignedInt(ty)) => ty_hint.checked_or(tcx.mk_mach_int(ty)),
                    // we had a type hint, so we can't have an unknown type
                    None => bug!(),
                };
                cx.eval(base, hint)?
            },
            Err(e) => return Err(e),
        };
        match cast_const(tcx, val, ety) {
            Ok(val) => val,
            Err(kind) => return Err(ConstEvalErr { span: e.span, kind: kind }),
        }
      }
      hir::ExprPath(ref qpath) => {
          let def = cx.tables.map(|tables| tables.qpath_def(qpath, e.id)).unwrap_or_else(|| {
            // There are no tables so we can only handle already-resolved HIR.
            match *qpath {
                hir::QPath::Resolved(_, ref path) => path.def,
                hir::QPath::TypeRelative(..) => Def::Err
            }
          });
          match def {
              Def::Const(def_id) |
              Def::AssociatedConst(def_id) => {
                  let substs = if let ExprTypeChecked = ty_hint {
                      Some(cx.tables.and_then(|tables| tables.node_id_item_substs(e.id))
                        .unwrap_or_else(|| tcx.intern_substs(&[])))
                  } else {
                      None
                  };
                  if let Some((expr, tables, ty)) = lookup_const_by_id(tcx, def_id, substs) {
                      let item_hint = match ty {
                          Some(ty) => ty_hint.checked_or(ty),
                          None => ty_hint,
                      };
                      let cx = ConstContext { tcx: tcx, tables: tables, fn_args: None };
                      match cx.eval(expr, item_hint) {
                          Ok(val) => val,
                          Err(err) => {
                              debug!("bad reference: {:?}, {:?}", err.description(), err.span);
                              signal!(e, ErroneousReferencedConstant(box err))
                          },
                      }
                  } else {
                      signal!(e, NonConstPath);
                  }
              },
              Def::VariantCtor(variant_def, ..) => {
                  if let Some((expr, tables)) = lookup_variant_by_id(tcx, variant_def) {
                      let cx = ConstContext { tcx: tcx, tables: tables, fn_args: None };
                      match cx.eval(expr, ty_hint) {
                          Ok(val) => val,
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
              Def::Method(id) | Def::Fn(id) => Function(id),
              Def::Err => signal!(e, UnresolvedPath),
              _ => signal!(e, NonConstPath),
          }
      }
      hir::ExprCall(ref callee, ref args) => {
          let sub_ty_hint = ty_hint.erase_hint();
          let callee_val = cx.eval(callee, sub_ty_hint)?;
          let did = match callee_val {
              Function(did) => did,
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
              let arg_hint = ty_hint.erase_hint();
              let arg_val = cx.eval(arg_expr, arg_hint)?;
              debug!("const call arg: {:?}", arg);
              if let Some(def_id) = arg {
                assert!(call_args.insert(def_id, arg_val).is_none());
              }
          }
          debug!("const call({:?})", call_args);
          let callee_cx = ConstContext {
            tcx: tcx,
            tables: tables,
            fn_args: Some(call_args)
          };
          callee_cx.eval(&body.value, ty_hint)?
      },
      hir::ExprLit(ref lit) => match lit_to_const(&lit.node, tcx, ety) {
          Ok(val) => val,
          Err(err) => signal!(e, err),
      },
      hir::ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => cx.eval(expr, ty_hint)?,
            None => signal!(e, UnimplementedConstVal("empty block")),
        }
      }
      hir::ExprType(ref e, _) => cx.eval(e, ty_hint)?,
      hir::ExprTup(ref fields) => {
        let field_hint = ty_hint.erase_hint();
        Tuple(fields.iter().map(|e| cx.eval(e, field_hint)).collect::<Result<_, _>>()?)
      }
      hir::ExprStruct(_, ref fields, _) => {
        let field_hint = ty_hint.erase_hint();
        Struct(fields.iter().map(|f| {
            cx.eval(&f.expr, field_hint).map(|v| (f.name.node, v))
        }).collect::<Result<_, _>>()?)
      }
      hir::ExprIndex(ref arr, ref idx) => {
        if !tcx.sess.features.borrow().const_indexing {
            signal!(e, IndexOpFeatureGated);
        }
        let arr_hint = ty_hint.erase_hint();
        let arr = cx.eval(arr, arr_hint)?;
        let idx_hint = ty_hint.checked_or(tcx.types.usize);
        let idx = match cx.eval(idx, idx_hint)? {
            Integral(Usize(i)) => i.as_u64(tcx.sess.target.uint_type),
            Integral(_) => bug!(),
            _ => signal!(idx, IndexNotInt),
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
        let elem_hint = ty_hint.erase_hint();
        Array(v.iter().map(|e| cx.eval(e, elem_hint)).collect::<Result<_, _>>()?)
      }
      hir::ExprRepeat(ref elem, count) => {
          let elem_hint = ty_hint.erase_hint();
          let len_hint = ty_hint.checked_or(tcx.types.usize);
          let n = if let Some(ty) = ety {
            // For cross-crate constants, we have the type already,
            // but not the body for `count`, so use the type.
            match ty.sty {
                ty::TyArray(_, n) => n as u64,
                _ => bug!()
            }
          } else {
            let n = &tcx.map.body(count).value;
            match ConstContext::new(tcx, count).eval(n, len_hint)? {
                Integral(Usize(i)) => i.as_u64(tcx.sess.target.uint_type),
                Integral(_) => signal!(e, RepeatCountNotNatural),
                _ => signal!(e, RepeatCountNotInt),
            }
          };
          Repeat(Box::new(cx.eval(elem, elem_hint)?), n)
      },
      hir::ExprTupField(ref base, index) => {
        let base_hint = ty_hint.erase_hint();
        let c = cx.eval(base, base_hint)?;
        if let Tuple(ref fields) = c {
            if let Some(elem) = fields.get(index.node) {
                elem.clone()
            } else {
                signal!(e, TupleIndexOutOfBounds);
            }
        } else {
            signal!(base, ExpectedConstTuple);
        }
      }
      hir::ExprField(ref base, field_name) => {
        let base_hint = ty_hint.erase_hint();
        let c = cx.eval(base, base_hint)?;
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

    match (ety.map(|t| &t.sty), result) {
        (Some(ref ty_hint), Integral(i)) => match infer(i, tcx, ty_hint) {
            Ok(inferred) => Ok(Integral(inferred)),
            Err(err) => signal!(e, err),
        },
        (_, result) => Ok(result),
    }
}

fn infer<'a, 'tcx>(i: ConstInt,
                   tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   ty_hint: &ty::TypeVariants<'tcx>)
                   -> Result<ConstInt, ErrKind> {
    use syntax::ast::*;

    match (ty_hint, i) {
        (&ty::TyInt(IntTy::I8), result @ I8(_)) => Ok(result),
        (&ty::TyInt(IntTy::I16), result @ I16(_)) => Ok(result),
        (&ty::TyInt(IntTy::I32), result @ I32(_)) => Ok(result),
        (&ty::TyInt(IntTy::I64), result @ I64(_)) => Ok(result),
        (&ty::TyInt(IntTy::I128), result @ I128(_)) => Ok(result),
        (&ty::TyInt(IntTy::Is), result @ Isize(_)) => Ok(result),

        (&ty::TyUint(UintTy::U8), result @ U8(_)) => Ok(result),
        (&ty::TyUint(UintTy::U16), result @ U16(_)) => Ok(result),
        (&ty::TyUint(UintTy::U32), result @ U32(_)) => Ok(result),
        (&ty::TyUint(UintTy::U64), result @ U64(_)) => Ok(result),
        (&ty::TyUint(UintTy::U128), result @ U128(_)) => Ok(result),
        (&ty::TyUint(UintTy::Us), result @ Usize(_)) => Ok(result),

        (&ty::TyInt(IntTy::I8), Infer(i)) => Ok(I8(i as i128 as i8)),
        (&ty::TyInt(IntTy::I16), Infer(i)) => Ok(I16(i as i128 as i16)),
        (&ty::TyInt(IntTy::I32), Infer(i)) => Ok(I32(i as i128 as i32)),
        (&ty::TyInt(IntTy::I64), Infer(i)) => Ok(I64(i as i128 as i64)),
        (&ty::TyInt(IntTy::I128), Infer(i)) => Ok(I128(i as i128)),
        (&ty::TyInt(IntTy::Is), Infer(i)) => {
            Ok(Isize(ConstIsize::new_truncating(i as i128, tcx.sess.target.int_type)))
        },

        (&ty::TyInt(IntTy::I8), InferSigned(i)) => Ok(I8(i as i8)),
        (&ty::TyInt(IntTy::I16), InferSigned(i)) => Ok(I16(i as i16)),
        (&ty::TyInt(IntTy::I32), InferSigned(i)) => Ok(I32(i as i32)),
        (&ty::TyInt(IntTy::I64), InferSigned(i)) => Ok(I64(i as i64)),
        (&ty::TyInt(IntTy::I128), InferSigned(i)) => Ok(I128(i)),
        (&ty::TyInt(IntTy::Is), InferSigned(i)) => {
            Ok(Isize(ConstIsize::new_truncating(i, tcx.sess.target.int_type)))
        },

        (&ty::TyUint(UintTy::U8), Infer(i)) => Ok(U8(i as u8)),
        (&ty::TyUint(UintTy::U16), Infer(i)) => Ok(U16(i as u16)),
        (&ty::TyUint(UintTy::U32), Infer(i)) => Ok(U32(i as u32)),
        (&ty::TyUint(UintTy::U64), Infer(i)) => Ok(U64(i as u64)),
        (&ty::TyUint(UintTy::U128), Infer(i)) => Ok(U128(i)),
        (&ty::TyUint(UintTy::Us), Infer(i)) => {
            Ok(Usize(ConstUsize::new_truncating(i, tcx.sess.target.uint_type)))
        },
        (&ty::TyUint(_), InferSigned(_)) => Err(IntermediateUnsignedNegative),

        (&ty::TyInt(ity), i) => Err(TypeMismatch(ity.to_string(), i)),
        (&ty::TyUint(ity), i) => Err(TypeMismatch(ity.to_string(), i)),

        (&ty::TyAdt(adt, _), i) if adt.is_enum() => {
            let hints = tcx.lookup_repr_hints(adt.did);
            let int_ty = tcx.enum_repr_type(hints.iter().next());
            infer(i, tcx, &int_ty.to_ty(tcx).sty)
        },
        (_, i) => Err(BadType(ConstVal::Integral(i))),
    }
}

fn resolve_trait_associated_const<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    trait_item_id: DefId,
    default_value: Option<(&'tcx Expr, Option<&'a ty::Tables<'tcx>>, Option<ty::Ty<'tcx>>)>,
    trait_id: DefId,
    rcvr_substs: &'tcx Substs<'tcx>
) -> Option<(&'tcx Expr, Option<&'a ty::Tables<'tcx>>, Option<ty::Ty<'tcx>>)>
{
    let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, rcvr_substs));
    debug!("resolve_trait_associated_const: trait_ref={:?}",
           trait_ref);

    tcx.populate_implementations_for_trait_if_necessary(trait_id);
    tcx.infer_ctxt((), Reveal::NotSpecializable).enter(|infcx| {
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
                    Some(ic) => lookup_const_by_id(tcx, ic.def_id, None),
                    None => default_value,
                }
            }
            _ => {
                bug!("resolve_trait_associated_const: unexpected vtable type")
            }
        }
    })
}

fn cast_const_int<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, val: ConstInt, ty: ty::Ty) -> CastResult {
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
        ty::TyFloat(ast::FloatTy::F64) => match val.erase_type() {
            Infer(u) => Ok(Float(F64(u as f64))),
            InferSigned(i) => Ok(Float(F64(i as f64))),
            _ => bug!("ConstInt::erase_type returned something other than Infer/InferSigned"),
        },
        ty::TyFloat(ast::FloatTy::F32) => match val.erase_type() {
            Infer(u) => Ok(Float(F32(u as f32))),
            InferSigned(i) => Ok(Float(F32(i as f32))),
            _ => bug!("ConstInt::erase_type returned something other than Infer/InferSigned"),
        },
        ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting an address to a raw ptr")),
        ty::TyChar => match infer(val, tcx, &ty::TyUint(ast::UintTy::U8)) {
            Ok(U8(u)) => Ok(Char(u as char)),
            // can only occur before typeck, typeck blocks `T as char` for `T` != `u8`
            _ => Err(CharCast(val)),
        },
        _ => Err(CannotCast),
    }
}

fn cast_const_float<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              val: ConstFloat,
                              ty: ty::Ty) -> CastResult {
    match ty.sty {
        ty::TyInt(_) | ty::TyUint(_) => {
            let i = match val {
                F32(f) if f >= 0.0 => Infer(f as u128),
                FInfer { f64: f, .. } |
                F64(f) if f >= 0.0 => Infer(f as u128),

                F32(f) => InferSigned(f as i128),
                FInfer { f64: f, .. } |
                F64(f) => InferSigned(f as i128)
            };

            if let (InferSigned(_), &ty::TyUint(_)) = (i, &ty.sty) {
                return Err(CannotCast);
            }

            cast_const_int(tcx, i, ty)
        }
        ty::TyFloat(ast::FloatTy::F64) => Ok(Float(F64(match val {
            F32(f) => f as f64,
            FInfer { f64: f, .. } | F64(f) => f
        }))),
        ty::TyFloat(ast::FloatTy::F32) => Ok(Float(F32(match val {
            F64(f) => f as f32,
            FInfer { f32: f, .. } | F32(f) => f
        }))),
        _ => Err(CannotCast),
    }
}

fn cast_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, val: ConstVal, ty: ty::Ty) -> CastResult {
    match val {
        Integral(i) => cast_const_int(tcx, i, ty),
        Bool(b) => cast_const_int(tcx, Infer(b as u128), ty),
        Float(f) => cast_const_float(tcx, f, ty),
        Char(c) => cast_const_int(tcx, Infer(c as u128), ty),
        Function(_) => Err(UnimplementedConstVal("casting fn pointers")),
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
                          ty_hint: Option<Ty<'tcx>>)
                          -> Result<ConstVal, ErrKind> {
    use syntax::ast::*;
    use syntax::ast::LitIntType::*;
    match *lit {
        LitKind::Str(ref s, _) => Ok(Str(s.as_str())),
        LitKind::ByteStr(ref data) => Ok(ByteStr(data.clone())),
        LitKind::Byte(n) => Ok(Integral(U8(n))),
        LitKind::Int(n, Signed(ity)) => {
            infer(InferSigned(n as i128), tcx, &ty::TyInt(ity)).map(Integral)
        },

        // FIXME: this should become u128.
        LitKind::Int(n, Unsuffixed) => {
            match ty_hint.map(|t| &t.sty) {
                Some(&ty::TyInt(ity)) => {
                    infer(InferSigned(n as i128), tcx, &ty::TyInt(ity)).map(Integral)
                },
                Some(&ty::TyUint(uty)) => {
                    infer(Infer(n as u128), tcx, &ty::TyUint(uty)).map(Integral)
                },
                None => Ok(Integral(Infer(n as u128))),
                Some(&ty::TyAdt(adt, _)) => {
                    let hints = tcx.lookup_repr_hints(adt.did);
                    let int_ty = tcx.enum_repr_type(hints.iter().next());
                    infer(Infer(n as u128), tcx, &int_ty.to_ty(tcx).sty).map(Integral)
                },
                Some(ty_hint) => bug!("bad ty_hint: {:?}, {:?}", ty_hint, lit),
            }
        },
        LitKind::Int(n, Unsigned(ity)) => {
            infer(Infer(n as u128), tcx, &ty::TyUint(ity)).map(Integral)
        },

        LitKind::Float(n, fty) => {
            parse_float(&n.as_str(), Some(fty)).map(Float)
        }
        LitKind::FloatUnsuffixed(n) => {
            let fty_hint = match ty_hint.map(|t| &t.sty) {
                Some(&ty::TyFloat(fty)) => Some(fty),
                _ => None
            };
            parse_float(&n.as_str(), fty_hint).map(Float)
        }
        LitKind::Bool(b) => Ok(Bool(b)),
        LitKind::Char(c) => Ok(Char(c)),
    }
}

fn parse_float(num: &str, fty_hint: Option<ast::FloatTy>)
               -> Result<ConstFloat, ErrKind> {
    let val = match fty_hint {
        Some(ast::FloatTy::F32) => num.parse::<f32>().map(F32),
        Some(ast::FloatTy::F64) => num.parse::<f64>().map(F64),
        None => {
            num.parse::<f32>().and_then(|f32| {
                num.parse::<f64>().map(|f64| {
                    FInfer { f32: f32, f64: f64 }
                })
            })
        }
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
        let a = match self.eval(a, ExprTypeChecked) {
            Ok(a) => a,
            Err(e) => {
                report_const_eval_err(tcx, &e, a.span, "expression").emit();
                return Err(ErrorReported);
            }
        };
        let b = match self.eval(b, ExprTypeChecked) {
            Ok(b) => b,
            Err(e) => {
                report_const_eval_err(tcx, &e, b.span, "expression").emit();
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
    let hint = UncheckedExprHint(tcx.types.usize);
    let count_expr = &tcx.map.body(count).value;
    match ConstContext::new(tcx, count).eval(count_expr, hint) {
        Ok(Integral(Usize(count))) => {
            let val = count.as_u64(tcx.sess.target.uint_type);
            assert_eq!(val as usize as u64, val);
            Ok(val as usize)
        },
        Ok(const_val) => {
            struct_span_err!(tcx.sess, count_expr.span, E0306,
                             "expected `usize` for {}, found {}",
                             reason,
                             const_val.description())
                .span_label(count_expr.span, &format!("expected `usize`"))
                .emit();

            Err(ErrorReported)
        }
        Err(err) => {
            let mut diag = report_const_eval_err(
                tcx, &err, count_expr.span, reason);

            if let hir::ExprPath(hir::QPath::Resolved(None, ref path)) = count_expr.node {
                if let Def::Local(..) = path.def {
                    diag.note(&format!("`{}` is a variable",
                                       tcx.map.node_to_pretty_string(count_expr.id)));
                }
            }

            diag.emit();
            Err(ErrorReported)
        }
    }
}
