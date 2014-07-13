// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]
#![allow(unsigned_negate)]

use metadata::csearch;
use middle::astencode;
use middle::def;
use middle::pat_util::def_to_path;
use middle::ty;
use middle::typeck::astconv;
use util::nodemap::{DefIdMap};

use syntax::ast::*;
use syntax::parse::token::InternedString;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::{ast, ast_map, ast_util};

use std::rc::Rc;
use std::gc::{Gc, GC};

//
// This pass classifies expressions by their constant-ness.
//
// Constant-ness comes in 3 flavours:
//
//   - Integer-constants: can be evaluated by the frontend all the way down
//     to their actual value. They are used in a few places (enum
//     discriminants, switch arms) and are a subset of
//     general-constants. They cover all the integer and integer-ish
//     literals (nil, bool, int, uint, char, iNN, uNN) and all integer
//     operators and copies applied to them.
//
//   - General-constants: can be evaluated by LLVM but not necessarily by
//     the frontend; usually due to reliance on target-specific stuff such
//     as "where in memory the value goes" or "what floating point mode the
//     target uses". This _includes_ integer-constants, plus the following
//     constructors:
//
//        fixed-size vectors and strings: [] and ""/_
//        vector and string slices: &[] and &""
//        tuples: (,)
//        enums: foo(...)
//        floating point literals and operators
//        & and * pointers
//        copies of general constants
//
//        (in theory, probably not at first: if/match on integer-const
//         conditions / descriminants)
//
//   - Non-constants: everything else.
//

pub enum constness {
    integral_const,
    general_const,
    non_const
}

type constness_cache = DefIdMap<constness>;

pub fn join(a: constness, b: constness) -> constness {
    match (a, b) {
      (integral_const, integral_const) => integral_const,
      (integral_const, general_const)
      | (general_const, integral_const)
      | (general_const, general_const) => general_const,
      _ => non_const
    }
}

pub fn join_all<It: Iterator<constness>>(mut cs: It) -> constness {
    cs.fold(integral_const, |a, b| join(a, b))
}

pub fn lookup_const(tcx: &ty::ctxt, e: &Expr) -> Option<Gc<Expr>> {
    let opt_def = tcx.def_map.borrow().find_copy(&e.id);
    match opt_def {
        Some(def::DefStatic(def_id, false)) => {
            lookup_const_by_id(tcx, def_id)
        }
        Some(def::DefVariant(enum_def, variant_def, _)) => {
            lookup_variant_by_id(tcx, enum_def, variant_def)
        }
        _ => None
    }
}

pub fn lookup_variant_by_id(tcx: &ty::ctxt,
                            enum_def: ast::DefId,
                            variant_def: ast::DefId)
                       -> Option<Gc<Expr>> {
    fn variant_expr(variants: &[ast::P<ast::Variant>],
                    id: ast::NodeId) -> Option<Gc<Expr>> {
        for variant in variants.iter() {
            if variant.node.id == id {
                return variant.node.disr_expr;
            }
        }
        None
    }

    if ast_util::is_local(enum_def) {
        {
            match tcx.map.find(enum_def.node) {
                None => None,
                Some(ast_map::NodeItem(it)) => match it.node {
                    ItemEnum(ast::EnumDef { variants: ref variants }, _) => {
                        variant_expr(variants.as_slice(), variant_def.node)
                    }
                    _ => None
                },
                Some(_) => None
            }
        }
    } else {
        match tcx.extern_const_variants.borrow().find(&variant_def) {
            Some(&e) => return e,
            None => {}
        }
        let e = match csearch::maybe_get_item_ast(tcx, enum_def,
            |a, b, c, d| astencode::decode_inlined_item(a, b, c, d)) {
            csearch::found(ast::IIItem(item)) => match item.node {
                ItemEnum(ast::EnumDef { variants: ref variants }, _) => {
                    variant_expr(variants.as_slice(), variant_def.node)
                }
                _ => None
            },
            _ => None
        };
        tcx.extern_const_variants.borrow_mut().insert(variant_def, e);
        return e;
    }
}

pub fn lookup_const_by_id(tcx: &ty::ctxt, def_id: ast::DefId)
                          -> Option<Gc<Expr>> {
    if ast_util::is_local(def_id) {
        {
            match tcx.map.find(def_id.node) {
                None => None,
                Some(ast_map::NodeItem(it)) => match it.node {
                    ItemStatic(_, ast::MutImmutable, const_expr) => {
                        Some(const_expr)
                    }
                    _ => None
                },
                Some(_) => None
            }
        }
    } else {
        match tcx.extern_const_statics.borrow().find(&def_id) {
            Some(&e) => return e,
            None => {}
        }
        let e = match csearch::maybe_get_item_ast(tcx, def_id,
            |a, b, c, d| astencode::decode_inlined_item(a, b, c, d)) {
            csearch::found(ast::IIItem(item)) => match item.node {
                ItemStatic(_, ast::MutImmutable, const_expr) => Some(const_expr),
                _ => None
            },
            _ => None
        };
        tcx.extern_const_statics.borrow_mut().insert(def_id, e);
        return e;
    }
}

struct ConstEvalVisitor<'a> {
    tcx: &'a ty::ctxt,
    ccache: constness_cache,
}

impl<'a> ConstEvalVisitor<'a> {
    fn classify(&mut self, e: &Expr) -> constness {
        let did = ast_util::local_def(e.id);
        match self.ccache.find(&did) {
            Some(&x) => return x,
            None => {}
        }
        let cn = match e.node {
            ast::ExprLit(ref lit) => {
                match lit.node {
                    ast::LitStr(..) | ast::LitFloat(..) => general_const,
                    _ => integral_const
                }
            }

            ast::ExprUnary(_, ref inner) | ast::ExprParen(ref inner) =>
                self.classify(&**inner),

            ast::ExprBinary(_, ref a, ref b) =>
                join(self.classify(&**a), self.classify(&**b)),

            ast::ExprTup(ref es) |
            ast::ExprVec(ref es) =>
                join_all(es.iter().map(|e| self.classify(&**e))),

            ast::ExprVstore(ref e, vstore) => {
                match vstore {
                    ast::ExprVstoreSlice => self.classify(&**e),
                    ast::ExprVstoreUniq |
                    ast::ExprVstoreMutSlice => non_const
                }
            }

            ast::ExprStruct(_, ref fs, None) => {
                let cs = fs.iter().map(|f| self.classify(&*f.expr));
                join_all(cs)
            }

            ast::ExprCast(ref base, _) => {
                let ty = ty::expr_ty(self.tcx, e);
                let base = self.classify(&**base);
                if ty::type_is_integral(ty) {
                    join(integral_const, base)
                } else if ty::type_is_fp(ty) {
                    join(general_const, base)
                } else {
                    non_const
                }
            }

            ast::ExprField(ref base, _, _) => self.classify(&**base),

            ast::ExprIndex(ref base, ref idx) =>
                join(self.classify(&**base), self.classify(&**idx)),

            ast::ExprAddrOf(ast::MutImmutable, ref base) =>
                self.classify(&**base),

            // FIXME: (#3728) we can probably do something CCI-ish
            // surrounding nonlocal constants. But we don't yet.
            ast::ExprPath(_) => self.lookup_constness(e),

            ast::ExprRepeat(..) => general_const,

            ast::ExprBlock(ref block) => {
                match block.expr {
                    Some(ref e) => self.classify(&**e),
                    None => integral_const
                }
            }

            _ => non_const
        };
        self.ccache.insert(did, cn);
        cn
    }

    fn lookup_constness(&self, e: &Expr) -> constness {
        match lookup_const(self.tcx, e) {
            Some(rhs) => {
                let ty = ty::expr_ty(self.tcx, &*rhs);
                if ty::type_is_integral(ty) {
                    integral_const
                } else {
                    general_const
                }
            }
            None => non_const
        }
    }

}

impl<'a> Visitor<()> for ConstEvalVisitor<'a> {
    fn visit_expr_post(&mut self, e: &Expr, _: ()) {
        self.classify(e);
    }
}

pub fn process_crate(krate: &ast::Crate,
                     tcx: &ty::ctxt) {
    let mut v = ConstEvalVisitor {
        tcx: tcx,
        ccache: DefIdMap::new(),
    };
    visit::walk_crate(&mut v, krate, ());
    tcx.sess.abort_if_errors();
}


// FIXME (#33): this doesn't handle big integer/float literals correctly
// (nor does the rest of our literal handling).
#[deriving(Clone, PartialEq)]
pub enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(InternedString),
    const_binary(Rc<Vec<u8> >),
    const_bool(bool),
    const_nil
}

pub fn const_expr_to_pat(tcx: &ty::ctxt, expr: Gc<Expr>) -> Gc<Pat> {
    let pat = match expr.node {
        ExprTup(ref exprs) =>
            PatTup(exprs.iter().map(|&expr| const_expr_to_pat(tcx, expr)).collect()),

        ExprCall(callee, ref args) => {
            let def = tcx.def_map.borrow().get_copy(&callee.id);
            tcx.def_map.borrow_mut().find_or_insert(expr.id, def);
            let path = match def {
                def::DefStruct(def_id) => def_to_path(tcx, def_id),
                def::DefVariant(_, variant_did, _) => def_to_path(tcx, variant_did),
                _ => unreachable!()
            };
            let pats = args.iter().map(|&expr| const_expr_to_pat(tcx, expr)).collect();
            PatEnum(path, Some(pats))
        }

        ExprStruct(ref path, ref fields, None) => {
            let field_pats = fields.iter().map(|field| FieldPat {
                ident: field.ident.node,
                pat: const_expr_to_pat(tcx, field.expr)
            }).collect();
            PatStruct(path.clone(), field_pats, false)
        }

        ExprVec(ref exprs) => {
            let pats = exprs.iter().map(|&expr| const_expr_to_pat(tcx, expr)).collect();
            PatVec(pats, None, vec![])
        }

        ExprPath(ref path) => {
            let opt_def = tcx.def_map.borrow().find_copy(&expr.id);
            match opt_def {
                Some(def::DefStruct(..)) =>
                    PatStruct(path.clone(), vec![], false),
                Some(def::DefVariant(..)) =>
                    PatEnum(path.clone(), None),
                _ => {
                    match lookup_const(tcx, &*expr) {
                        Some(actual) => return const_expr_to_pat(tcx, actual),
                        _ => unreachable!()
                    }
                }
            }
        }

        _ => PatLit(expr)
    };
    box (GC) Pat { id: expr.id, node: pat, span: expr.span }
}

pub fn eval_const_expr(tcx: &ty::ctxt, e: &Expr) -> const_val {
    match eval_const_expr_partial(tcx, e) {
        Ok(r) => r,
        Err(s) => tcx.sess.span_fatal(e.span, s.as_slice())
    }
}

pub fn eval_const_expr_partial<T: ty::ExprTyProvider>(tcx: &T, e: &Expr)
                            -> Result<const_val, String> {
    fn fromb(b: bool) -> Result<const_val, String> { Ok(const_int(b as i64)) }
    match e.node {
      ExprUnary(UnNeg, ref inner) => {
        match eval_const_expr_partial(tcx, &**inner) {
          Ok(const_float(f)) => Ok(const_float(-f)),
          Ok(const_int(i)) => Ok(const_int(-i)),
          Ok(const_uint(i)) => Ok(const_uint(-i)),
          Ok(const_str(_)) => Err("negate on string".to_string()),
          Ok(const_bool(_)) => Err("negate on boolean".to_string()),
          ref err => ((*err).clone())
        }
      }
      ExprUnary(UnNot, ref inner) => {
        match eval_const_expr_partial(tcx, &**inner) {
          Ok(const_int(i)) => Ok(const_int(!i)),
          Ok(const_uint(i)) => Ok(const_uint(!i)),
          Ok(const_bool(b)) => Ok(const_bool(!b)),
          _ => Err("not on float or string".to_string())
        }
      }
      ExprBinary(op, ref a, ref b) => {
        match (eval_const_expr_partial(tcx, &**a),
               eval_const_expr_partial(tcx, &**b)) {
          (Ok(const_float(a)), Ok(const_float(b))) => {
            match op {
              BiAdd => Ok(const_float(a + b)),
              BiSub => Ok(const_float(a - b)),
              BiMul => Ok(const_float(a * b)),
              BiDiv => Ok(const_float(a / b)),
              BiRem => Ok(const_float(a % b)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b),
              _ => Err("can't do this op on floats".to_string())
            }
          }
          (Ok(const_int(a)), Ok(const_int(b))) => {
            match op {
              BiAdd => Ok(const_int(a + b)),
              BiSub => Ok(const_int(a - b)),
              BiMul => Ok(const_int(a * b)),
              BiDiv if b == 0 => {
                  Err("attempted to divide by zero".to_string())
              }
              BiDiv => Ok(const_int(a / b)),
              BiRem if b == 0 => {
                  Err("attempted remainder with a divisor of \
                       zero".to_string())
              }
              BiRem => Ok(const_int(a % b)),
              BiAnd | BiBitAnd => Ok(const_int(a & b)),
              BiOr | BiBitOr => Ok(const_int(a | b)),
              BiBitXor => Ok(const_int(a ^ b)),
              BiShl => Ok(const_int(a << b as uint)),
              BiShr => Ok(const_int(a >> b as uint)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b)
            }
          }
          (Ok(const_uint(a)), Ok(const_uint(b))) => {
            match op {
              BiAdd => Ok(const_uint(a + b)),
              BiSub => Ok(const_uint(a - b)),
              BiMul => Ok(const_uint(a * b)),
              BiDiv if b == 0 => {
                  Err("attempted to divide by zero".to_string())
              }
              BiDiv => Ok(const_uint(a / b)),
              BiRem if b == 0 => {
                  Err("attempted remainder with a divisor of \
                       zero".to_string())
              }
              BiRem => Ok(const_uint(a % b)),
              BiAnd | BiBitAnd => Ok(const_uint(a & b)),
              BiOr | BiBitOr => Ok(const_uint(a | b)),
              BiBitXor => Ok(const_uint(a ^ b)),
              BiShl => Ok(const_uint(a << b as uint)),
              BiShr => Ok(const_uint(a >> b as uint)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (Ok(const_int(a)), Ok(const_uint(b))) => {
            match op {
              BiShl => Ok(const_int(a << b as uint)),
              BiShr => Ok(const_int(a >> b as uint)),
              _ => Err("can't do this op on an int and uint".to_string())
            }
          }
          (Ok(const_uint(a)), Ok(const_int(b))) => {
            match op {
              BiShl => Ok(const_uint(a << b as uint)),
              BiShr => Ok(const_uint(a >> b as uint)),
              _ => Err("can't do this op on a uint and int".to_string())
            }
          }
          (Ok(const_bool(a)), Ok(const_bool(b))) => {
            Ok(const_bool(match op {
              BiAnd => a && b,
              BiOr => a || b,
              BiBitXor => a ^ b,
              BiBitAnd => a & b,
              BiBitOr => a | b,
              BiEq => a == b,
              BiNe => a != b,
              _ => return Err("can't do this op on bools".to_string())
             }))
          }
          _ => Err("bad operands for binary".to_string())
        }
      }
      ExprCast(ref base, ref target_ty) => {
        // This tends to get called w/o the type actually having been
        // populated in the ctxt, which was causing things to blow up
        // (#5900). Fall back to doing a limited lookup to get past it.
        let ety = ty::expr_ty_opt(tcx.ty_ctxt(), e)
                .or_else(|| astconv::ast_ty_to_prim_ty(tcx.ty_ctxt(), &**target_ty))
                .unwrap_or_else(|| {
                    tcx.ty_ctxt().sess.span_fatal(target_ty.span,
                                                  "target type not found for \
                                                   const cast")
                });

        let base = eval_const_expr_partial(tcx, &**base);
        match base {
            Err(_) => base,
            Ok(val) => {
                match ty::get(ety).sty {
                    ty::ty_float(_) => {
                        match val {
                            const_uint(u) => Ok(const_float(u as f64)),
                            const_int(i) => Ok(const_float(i as f64)),
                            const_float(f) => Ok(const_float(f)),
                            _ => Err("can't cast float to str".to_string()),
                        }
                    }
                    ty::ty_uint(_) => {
                        match val {
                            const_uint(u) => Ok(const_uint(u)),
                            const_int(i) => Ok(const_uint(i as u64)),
                            const_float(f) => Ok(const_uint(f as u64)),
                            _ => Err("can't cast str to uint".to_string()),
                        }
                    }
                    ty::ty_int(_) | ty::ty_bool => {
                        match val {
                            const_uint(u) => Ok(const_int(u as i64)),
                            const_int(i) => Ok(const_int(i)),
                            const_float(f) => Ok(const_int(f as i64)),
                            _ => Err("can't cast str to int".to_string()),
                        }
                    }
                    _ => Err("can't cast this type".to_string())
                }
            }
        }
      }
      ExprPath(_) => {
          match lookup_const(tcx.ty_ctxt(), e) {
              Some(actual_e) => eval_const_expr_partial(tcx.ty_ctxt(), &*actual_e),
              None => Err("non-constant path in constant expr".to_string())
          }
      }
      ExprLit(ref lit) => Ok(lit_to_const(&**lit)),
      // If we have a vstore, just keep going; it has to be a string
      ExprVstore(ref e, _) => eval_const_expr_partial(tcx, &**e),
      ExprParen(ref e)     => eval_const_expr_partial(tcx, &**e),
      ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => eval_const_expr_partial(tcx, &**expr),
            None => Ok(const_int(0i64))
        }
      }
      _ => Err("unsupported constant expr".to_string())
    }
}

pub fn lit_to_const(lit: &Lit) -> const_val {
    match lit.node {
        LitStr(ref s, _) => const_str((*s).clone()),
        LitBinary(ref data) => {
            const_binary(Rc::new(data.iter().map(|x| *x).collect()))
        }
        LitByte(n) => const_uint(n as u64),
        LitChar(n) => const_uint(n as u64),
        LitInt(n, _) => const_int(n),
        LitUint(n, _) => const_uint(n),
        LitIntUnsuffixed(n) => const_int(n),
        LitFloat(ref n, _) | LitFloatUnsuffixed(ref n) => {
            const_float(from_str::<f64>(n.get()).unwrap() as f64)
        }
        LitNil => const_nil,
        LitBool(b) => const_bool(b)
    }
}

fn compare_vals<T: PartialOrd>(a: T, b: T) -> Option<int> {
    Some(if a == b { 0 } else if a < b { -1 } else { 1 })
}
pub fn compare_const_vals(a: &const_val, b: &const_val) -> Option<int> {
    match (a, b) {
        (&const_int(a), &const_int(b)) => compare_vals(a, b),
        (&const_uint(a), &const_uint(b)) => compare_vals(a, b),
        (&const_float(a), &const_float(b)) => compare_vals(a, b),
        (&const_str(ref a), &const_str(ref b)) => compare_vals(a, b),
        (&const_bool(a), &const_bool(b)) => compare_vals(a, b),
        (&const_binary(ref a), &const_binary(ref b)) => compare_vals(a, b),
        (&const_nil, &const_nil) => compare_vals((), ()),
        _ => None
    }
}

pub fn compare_lit_exprs(tcx: &ty::ctxt, a: &Expr, b: &Expr) -> Option<int> {
    compare_const_vals(&eval_const_expr(tcx, a), &eval_const_expr(tcx, b))
}
