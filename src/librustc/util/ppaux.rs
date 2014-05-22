// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty::{ReSkolemized, ReVar};
use middle::ty::{BoundRegion, BrAnon, BrNamed};
use middle::ty::{BrFresh, ctxt};
use middle::ty::{mt, t, param_ty};
use middle::ty::{ReFree, ReScope, ReInfer, ReStatic, Region,
                 ReEmpty};
use middle::ty::{ty_bool, ty_char, ty_bot, ty_box, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_str, ty_vec, ty_float, ty_bare_fn, ty_closure};
use middle::ty::{ty_nil, ty_param, ty_ptr, ty_rptr, ty_self, ty_tup};
use middle::ty::{ty_uniq, ty_trait, ty_int, ty_uint, ty_infer};
use middle::ty;
use middle::typeck;

use std::rc::Rc;
use std::strbuf::StrBuf;
use syntax::abi;
use syntax::ast_map;
use syntax::codemap::{Span, Pos};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::{ast, ast_util};
use syntax::owned_slice::OwnedSlice;

/// Produces a string suitable for debugging output.
pub trait Repr {
    fn repr(&self, tcx: &ctxt) -> StrBuf;
}

/// Produces a string suitable for showing to the user.
pub trait UserString {
    fn user_string(&self, tcx: &ctxt) -> StrBuf;
}

pub fn note_and_explain_region(cx: &ctxt,
                               prefix: &str,
                               region: ty::Region,
                               suffix: &str) {
    match explain_region_and_span(cx, region) {
      (ref str, Some(span)) => {
        cx.sess.span_note(
            span,
            format!("{}{}{}", prefix, *str, suffix).as_slice());
      }
      (ref str, None) => {
        cx.sess.note(
            format!("{}{}{}", prefix, *str, suffix).as_slice());
      }
    }
}

pub fn explain_region_and_span(cx: &ctxt, region: ty::Region)
                            -> (StrBuf, Option<Span>) {
    return match region {
      ReScope(node_id) => {
        match cx.map.find(node_id) {
          Some(ast_map::NodeBlock(ref blk)) => {
            explain_span(cx, "block", blk.span)
          }
          Some(ast_map::NodeExpr(expr)) => {
            match expr.node {
              ast::ExprCall(..) => explain_span(cx, "call", expr.span),
              ast::ExprMethodCall(..) => {
                explain_span(cx, "method call", expr.span)
              },
              ast::ExprMatch(..) => explain_span(cx, "match", expr.span),
              _ => explain_span(cx, "expression", expr.span)
            }
          }
          Some(ast_map::NodeStmt(stmt)) => {
              explain_span(cx, "statement", stmt.span)
          }
          Some(ast_map::NodeItem(it)) if (match it.node {
                ast::ItemFn(..) => true, _ => false}) => {
              explain_span(cx, "function body", it.span)
          }
          Some(_) | None => {
            // this really should not happen
            (format_strbuf!("unknown scope: {}.  Please report a bug.",
                            node_id),
             None)
          }
        }
      }

      ReFree(ref fr) => {
        let prefix = match fr.bound_region {
          BrAnon(idx) => {
              format_strbuf!("the anonymous lifetime \\#{} defined on",
                             idx + 1)
          }
          BrFresh(_) => "an anonymous lifetime defined on".to_strbuf(),
          _ => {
              format_strbuf!("the lifetime {} as defined on",
                             bound_region_ptr_to_str(cx, fr.bound_region))
          }
        };

        match cx.map.find(fr.scope_id) {
          Some(ast_map::NodeBlock(ref blk)) => {
            let (msg, opt_span) = explain_span(cx, "block", blk.span);
            (format_strbuf!("{} {}", prefix, msg), opt_span)
          }
          Some(ast_map::NodeItem(it)) if match it.node {
                ast::ItemImpl(..) => true, _ => false} => {
            let (msg, opt_span) = explain_span(cx, "impl", it.span);
            (format_strbuf!("{} {}", prefix, msg), opt_span)
          }
          Some(_) | None => {
            // this really should not happen
            (format_strbuf!("{} node {}", prefix, fr.scope_id), None)
          }
        }
      }

      ReStatic => { ("the static lifetime".to_strbuf(), None) }

      ReEmpty => { ("the empty lifetime".to_strbuf(), None) }

      // I believe these cases should not occur (except when debugging,
      // perhaps)
      ty::ReInfer(_) | ty::ReEarlyBound(..) | ty::ReLateBound(..) => {
        (format_strbuf!("lifetime {:?}", region), None)
      }
    };

    fn explain_span(cx: &ctxt, heading: &str, span: Span)
        -> (StrBuf, Option<Span>) {
        let lo = cx.sess.codemap().lookup_char_pos_adj(span.lo);
        (format_strbuf!("the {} at {}:{}",
                        heading,
                        lo.line,
                        lo.col.to_uint()), Some(span))
    }
}

pub fn bound_region_ptr_to_str(cx: &ctxt, br: BoundRegion) -> StrBuf {
    bound_region_to_str(cx, "&", true, br)
}

pub fn bound_region_to_str(cx: &ctxt,
                           prefix: &str, space: bool,
                           br: BoundRegion) -> StrBuf {
    let space_str = if space { " " } else { "" };

    if cx.sess.verbose() {
        return format_strbuf!("{}{}{}", prefix, br.repr(cx), space_str)
    }

    match br {
        BrNamed(_, name) => {
            format_strbuf!("{}'{}{}",
                           prefix,
                           token::get_name(name),
                           space_str)
        }
        BrAnon(_) => prefix.to_strbuf(),
        BrFresh(_) => prefix.to_strbuf(),
    }
}

// In general, if you are giving a region error message,
// you should use `explain_region()` or, better yet,
// `note_and_explain_region()`
pub fn region_ptr_to_str(cx: &ctxt, region: Region) -> StrBuf {
    region_to_str(cx, "&", true, region)
}

pub fn region_to_str(cx: &ctxt, prefix: &str, space: bool, region: Region) -> StrBuf {
    let space_str = if space { " " } else { "" };

    if cx.sess.verbose() {
        return format_strbuf!("{}{}{}", prefix, region.repr(cx), space_str)
    }

    // These printouts are concise.  They do not contain all the information
    // the user might want to diagnose an error, but there is basically no way
    // to fit that into a short string.  Hence the recommendation to use
    // `explain_region()` or `note_and_explain_region()`.
    match region {
        ty::ReScope(_) => prefix.to_strbuf(),
        ty::ReEarlyBound(_, _, name) => {
            token::get_name(name).get().to_strbuf()
        }
        ty::ReLateBound(_, br) => bound_region_to_str(cx, prefix, space, br),
        ty::ReFree(ref fr) => bound_region_to_str(cx, prefix, space, fr.bound_region),
        ty::ReInfer(ReSkolemized(_, br)) => {
            bound_region_to_str(cx, prefix, space, br)
        }
        ty::ReInfer(ReVar(_)) => prefix.to_strbuf(),
        ty::ReStatic => format_strbuf!("{}'static{}", prefix, space_str),
        ty::ReEmpty => format_strbuf!("{}'<empty>{}", prefix, space_str),
    }
}

pub fn mutability_to_str(m: ast::Mutability) -> StrBuf {
    match m {
        ast::MutMutable => "mut ".to_strbuf(),
        ast::MutImmutable => "".to_strbuf(),
    }
}

pub fn mt_to_str(cx: &ctxt, m: &mt) -> StrBuf {
    format_strbuf!("{}{}", mutability_to_str(m.mutbl), ty_to_str(cx, m.ty))
}

pub fn trait_store_to_str(cx: &ctxt, s: ty::TraitStore) -> StrBuf {
    match s {
        ty::UniqTraitStore => "Box ".to_strbuf(),
        ty::RegionTraitStore(r, m) => {
            format_strbuf!("{}{}",
                           region_ptr_to_str(cx, r),
                           mutability_to_str(m))
        }
    }
}

pub fn vec_map_to_str<T>(ts: &[T], f: |t: &T| -> StrBuf) -> StrBuf {
    let tstrs = ts.iter().map(f).collect::<Vec<StrBuf>>();
    format_strbuf!("[{}]", tstrs.connect(", "))
}

pub fn fn_sig_to_str(cx: &ctxt, typ: &ty::FnSig) -> StrBuf {
    format_strbuf!("fn{}{} -> {}",
                   typ.binder_id,
                   typ.inputs.repr(cx),
                   typ.output.repr(cx))
}

pub fn trait_ref_to_str(cx: &ctxt, trait_ref: &ty::TraitRef) -> StrBuf {
    trait_ref.user_string(cx).to_strbuf()
}

pub fn ty_to_str(cx: &ctxt, typ: t) -> StrBuf {
    fn fn_input_to_str(cx: &ctxt, input: ty::t) -> StrBuf {
        ty_to_str(cx, input).to_strbuf()
    }
    fn bare_fn_to_str(cx: &ctxt,
                      fn_style: ast::FnStyle,
                      abi: abi::Abi,
                      ident: Option<ast::Ident>,
                      sig: &ty::FnSig)
                      -> StrBuf {
        let mut s = StrBuf::new();
        match fn_style {
            ast::NormalFn => {}
            _ => {
                s.push_str(fn_style.to_str().as_slice());
                s.push_char(' ');
            }
        };

        if abi != abi::Rust {
            s.push_str(format!("extern {} ", abi.to_str()).as_slice());
        };

        s.push_str("fn");

        match ident {
            Some(i) => {
                s.push_char(' ');
                s.push_str(token::get_ident(i).get());
            }
            _ => { }
        }

        push_sig_to_str(cx, &mut s, '(', ')', sig);

        s
    }

    fn closure_to_str(cx: &ctxt, cty: &ty::ClosureTy) -> StrBuf {
        let mut s = StrBuf::new();

        match cty.store {
            ty::UniqTraitStore => {}
            ty::RegionTraitStore(region, _) => {
                s.push_str(region_to_str(cx, "", true, region).as_slice());
            }
        }

        match cty.fn_style {
            ast::NormalFn => {}
            _ => {
                s.push_str(cty.fn_style.to_str().as_slice());
                s.push_char(' ');
            }
        };

        match cty.store {
            ty::UniqTraitStore => {
                assert_eq!(cty.onceness, ast::Once);
                s.push_str("proc");
                push_sig_to_str(cx, &mut s, '(', ')', &cty.sig);
            }
            ty::RegionTraitStore(..) => {
                match cty.onceness {
                    ast::Many => {}
                    ast::Once => s.push_str("once ")
                }
                push_sig_to_str(cx, &mut s, '|', '|', &cty.sig);
            }
        }

        if !cty.bounds.is_empty() {
            s.push_str(":");
            s.push_str(cty.bounds.repr(cx).as_slice());
        }

        s
    }

    fn push_sig_to_str(cx: &ctxt,
                       s: &mut StrBuf,
                       bra: char,
                       ket: char,
                       sig: &ty::FnSig) {
        s.push_char(bra);
        let strs: Vec<StrBuf> = sig.inputs.iter().map(|a| fn_input_to_str(cx, *a)).collect();
        s.push_str(strs.connect(", ").as_slice());
        if sig.variadic {
            s.push_str(", ...");
        }
        s.push_char(ket);

        if ty::get(sig.output).sty != ty_nil {
            s.push_str(" -> ");
            if ty::type_is_bot(sig.output) {
                s.push_char('!');
            } else {
                s.push_str(ty_to_str(cx, sig.output).as_slice());
            }
        }
    }

    // if there is an id, print that instead of the structural type:
    /*for def_id in ty::type_def_id(typ).iter() {
        // note that this typedef cannot have type parameters
        return ty::item_path_str(cx, *def_id);
    }*/

    // pretty print the structural type representation:
    return match ty::get(typ).sty {
      ty_nil => "()".to_strbuf(),
      ty_bot => "!".to_strbuf(),
      ty_bool => "bool".to_strbuf(),
      ty_char => "char".to_strbuf(),
      ty_int(t) => ast_util::int_ty_to_str(t, None,
                                           ast_util::AutoSuffix).to_strbuf(),
      ty_uint(t) => ast_util::uint_ty_to_str(t, None,
                                             ast_util::AutoSuffix).to_strbuf(),
      ty_float(t) => ast_util::float_ty_to_str(t).to_strbuf(),
      ty_box(typ) => format_strbuf!("@{}", ty_to_str(cx, typ)),
      ty_uniq(typ) => format_strbuf!("~{}", ty_to_str(cx, typ)),
      ty_ptr(ref tm) => format_strbuf!("*{}", mt_to_str(cx, tm)),
      ty_rptr(r, ref tm) => {
          let mut buf = region_ptr_to_str(cx, r);
          buf.push_str(mt_to_str(cx, tm).as_slice());
          buf
      }
      ty_tup(ref elems) => {
        let strs: Vec<StrBuf> = elems.iter().map(|elem| ty_to_str(cx, *elem)).collect();
        format_strbuf!("({})", strs.connect(","))
      }
      ty_closure(ref f) => {
          closure_to_str(cx, *f)
      }
      ty_bare_fn(ref f) => {
          bare_fn_to_str(cx, f.fn_style, f.abi, None, &f.sig)
      }
      ty_infer(infer_ty) => infer_ty.to_str().to_strbuf(),
      ty_err => "[type error]".to_strbuf(),
      ty_param(param_ty {idx: id, def_id: did}) => {
          let ident = match cx.ty_param_defs.borrow().find(&did.node) {
              Some(def) => token::get_ident(def.ident).get().to_strbuf(),
              // This can only happen when a type mismatch error happens and
              // the actual type has more type parameters than the expected one.
              None => format_strbuf!("<generic \\#{}>", id)
          };
          if !cx.sess.verbose() {
              ident
          } else {
              format_strbuf!("{}:{:?}", ident, did)
          }
      }
      ty_self(..) => "Self".to_strbuf(),
      ty_enum(did, ref substs) | ty_struct(did, ref substs) => {
        let base = ty::item_path_str(cx, did);
        parameterized(cx,
                      base.as_slice(),
                      &substs.regions,
                      substs.tps.as_slice(),
                      did,
                      false)
      }
      ty_trait(box ty::TyTrait {
          def_id: did, ref substs, store, ref bounds
      }) => {
        let base = ty::item_path_str(cx, did);
        let ty = parameterized(cx, base.as_slice(), &substs.regions,
                               substs.tps.as_slice(), did, true);
        let bound_sep = if bounds.is_empty() { "" } else { ":" };
        let bound_str = bounds.repr(cx);
        format_strbuf!("{}{}{}{}",
                       trait_store_to_str(cx, store),
                       ty,
                       bound_sep,
                       bound_str)
      }
      ty_str => "str".to_strbuf(),
      ty_vec(ref mt, sz) => {
          match sz {
              Some(n) => {
                  format_strbuf!("[{}, .. {}]", mt_to_str(cx, mt), n)
              }
              None => format_strbuf!("[{}]", ty_to_str(cx, mt.ty)),
          }
      }
    }
}

pub fn parameterized(cx: &ctxt,
                     base: &str,
                     regions: &ty::RegionSubsts,
                     tps: &[ty::t],
                     did: ast::DefId,
                     is_trait: bool)
                     -> StrBuf {
    let mut strs = Vec::new();
    match *regions {
        ty::ErasedRegions => { }
        ty::NonerasedRegions(ref regions) => {
            for &r in regions.iter() {
                strs.push(region_to_str(cx, "", false, r))
            }
        }
    }

    let generics = if is_trait {
        ty::lookup_trait_def(cx, did).generics.clone()
    } else {
        ty::lookup_item_type(cx, did).generics
    };
    let ty_params = generics.type_param_defs();
    let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
    let num_defaults = if has_defaults {
        // We should have a borrowed version of substs instead of cloning.
        let mut substs = ty::substs {
            tps: Vec::from_slice(tps),
            regions: regions.clone(),
            self_ty: None
        };
        ty_params.iter().zip(tps.iter()).rev().take_while(|&(def, &actual)| {
            substs.tps.pop();
            match def.default {
                Some(default) => ty::subst(cx, &substs, default) == actual,
                None => false
            }
        }).len()
    } else {
        0
    };

    for t in tps.slice_to(tps.len() - num_defaults).iter() {
        strs.push(ty_to_str(cx, *t))
    }

    if strs.len() > 0u {
        format_strbuf!("{}<{}>", base, strs.connect(","))
    } else {
        format_strbuf!("{}", base)
    }
}

pub fn ty_to_short_str(cx: &ctxt, typ: t) -> StrBuf {
    let mut s = typ.repr(cx).to_strbuf();
    if s.len() >= 32u {
        s = s.as_slice().slice(0u, 32u).to_strbuf();
    }
    return s;
}

impl<T:Repr> Repr for Option<T> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match self {
            &None => "None".to_strbuf(),
            &Some(ref t) => t.repr(tcx),
        }
    }
}

impl<T:Repr,U:Repr> Repr for Result<T,U> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match self {
            &Ok(ref t) => t.repr(tcx),
            &Err(ref u) => format_strbuf!("Err({})", u.repr(tcx))
        }
    }
}

impl Repr for () {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        "()".to_strbuf()
    }
}

impl<T:Repr> Repr for Rc<T> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        (&**self).repr(tcx)
    }
}

impl<T:Repr> Repr for @T {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        (&**self).repr(tcx)
    }
}

impl<T:Repr> Repr for Box<T> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        (&**self).repr(tcx)
    }
}

fn repr_vec<T:Repr>(tcx: &ctxt, v: &[T]) -> StrBuf {
    vec_map_to_str(v, |t| t.repr(tcx))
}

impl<'a, T:Repr> Repr for &'a [T] {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        repr_vec(tcx, *self)
    }
}

impl<T:Repr> Repr for OwnedSlice<T> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        repr_vec(tcx, self.as_slice())
    }
}

// This is necessary to handle types like Option<~[T]>, for which
// autoderef cannot convert the &[T] handler
impl<T:Repr> Repr for Vec<T> {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        repr_vec(tcx, self.as_slice())
    }
}

impl Repr for ty::TypeParameterDef {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("TypeParameterDef({:?}, {})",
                       self.def_id,
                       self.bounds.repr(tcx))
    }
}

impl Repr for ty::RegionParameterDef {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("RegionParameterDef({}, {:?})",
                       token::get_name(self.name),
                       self.def_id)
    }
}

impl Repr for ty::t {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        ty_to_str(tcx, *self)
    }
}

impl Repr for ty::substs {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("substs(regions={}, self_ty={}, tps={})",
                       self.regions.repr(tcx),
                       self.self_ty.repr(tcx),
                       self.tps.repr(tcx))
    }
}

impl Repr for ty::ItemSubsts {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("ItemSubsts({})", self.substs.repr(tcx))
    }
}

impl Repr for ty::RegionSubsts {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match *self {
            ty::ErasedRegions => "erased".to_strbuf(),
            ty::NonerasedRegions(ref regions) => regions.repr(tcx)
        }
    }
}

impl Repr for ty::ParamBounds {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        let mut res = Vec::new();
        for b in self.builtin_bounds.iter() {
            res.push(match b {
                ty::BoundStatic => "'static".to_strbuf(),
                ty::BoundSend => "Send".to_strbuf(),
                ty::BoundSized => "Sized".to_strbuf(),
                ty::BoundCopy => "Copy".to_strbuf(),
                ty::BoundShare => "Share".to_strbuf(),
            });
        }
        for t in self.trait_bounds.iter() {
            res.push(t.repr(tcx));
        }
        res.connect("+").to_strbuf()
    }
}

impl Repr for ty::TraitRef {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        trait_ref_to_str(tcx, self)
    }
}

impl Repr for ast::Expr {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("expr({}: {})", self.id, pprust::expr_to_str(self))
    }
}

impl Repr for ast::Item {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("item({})", tcx.map.node_to_str(self.id))
    }
}

impl Repr for ast::Stmt {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("stmt({}: {})",
                       ast_util::stmt_id(self),
                       pprust::stmt_to_str(self))
    }
}

impl Repr for ast::Pat {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("pat({}: {})", self.id, pprust::pat_to_str(self))
    }
}

impl Repr for ty::BoundRegion {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match *self {
            ty::BrAnon(id) => format_strbuf!("BrAnon({})", id),
            ty::BrNamed(id, name) => {
                format_strbuf!("BrNamed({}, {})",
                               id.repr(tcx),
                               token::get_name(name))
            }
            ty::BrFresh(id) => format_strbuf!("BrFresh({})", id),
        }
    }
}

impl Repr for ty::Region {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match *self {
            ty::ReEarlyBound(id, index, name) => {
                format_strbuf!("ReEarlyBound({}, {}, {})",
                               id,
                               index,
                               token::get_name(name))
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                format_strbuf!("ReLateBound({}, {})",
                               binder_id,
                               bound_region.repr(tcx))
            }

            ty::ReFree(ref fr) => {
                format_strbuf!("ReFree({}, {})",
                               fr.scope_id,
                               fr.bound_region.repr(tcx))
            }

            ty::ReScope(id) => {
                format_strbuf!("ReScope({})", id)
            }

            ty::ReStatic => {
                "ReStatic".to_strbuf()
            }

            ty::ReInfer(ReVar(ref vid)) => {
                format_strbuf!("ReInfer({})", vid.id)
            }

            ty::ReInfer(ReSkolemized(id, ref bound_region)) => {
                format_strbuf!("re_skolemized({}, {})",
                               id,
                               bound_region.repr(tcx))
            }

            ty::ReEmpty => {
                "ReEmpty".to_strbuf()
            }
        }
    }
}

impl Repr for ast::DefId {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.krate == ast::LOCAL_CRATE {
            {
                match tcx.map.find(self.node) {
                    Some(ast_map::NodeItem(..)) |
                    Some(ast_map::NodeForeignItem(..)) |
                    Some(ast_map::NodeMethod(..)) |
                    Some(ast_map::NodeTraitMethod(..)) |
                    Some(ast_map::NodeVariant(..)) |
                    Some(ast_map::NodeStructCtor(..)) => {
                        return format_strbuf!(
                                "{:?}:{}",
                                *self,
                                ty::item_path_str(tcx, *self))
                    }
                    _ => {}
                }
            }
        }
        return format_strbuf!("{:?}", *self)
    }
}

impl Repr for ty::ty_param_bounds_and_ty {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("ty_param_bounds_and_ty \\{generics: {}, ty: {}\\}",
                       self.generics.repr(tcx),
                       self.ty.repr(tcx))
    }
}

impl Repr for ty::Generics {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("Generics(type_param_defs: {}, \
                           region_param_defs: {})",
                 self.type_param_defs().repr(tcx),
                 self.region_param_defs().repr(tcx))
    }
}

impl Repr for ty::ItemVariances {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("IterVariances(self_param={}, \
                                type_params={}, \
                                region_params={})",
                 self.self_param.repr(tcx),
                 self.type_params.repr(tcx),
                 self.region_params.repr(tcx))
    }
}

impl Repr for ty::Variance {
    fn repr(&self, _: &ctxt) -> StrBuf {
        self.to_str().to_strbuf()
    }
}

impl Repr for ty::Method {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("method(ident: {}, generics: {}, fty: {}, \
                        explicit_self: {}, vis: {}, def_id: {})",
                       self.ident.repr(tcx),
                       self.generics.repr(tcx),
                       self.fty.repr(tcx),
                       self.explicit_self.repr(tcx),
                       self.vis.repr(tcx),
                       self.def_id.repr(tcx))
    }
}

impl Repr for ast::Name {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        token::get_name(*self).get().to_strbuf()
    }
}

impl Repr for ast::Ident {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        token::get_ident(*self).get().to_strbuf()
    }
}

impl Repr for ast::ExplicitSelf_ {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl Repr for ast::Visibility {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl Repr for ty::BareFnTy {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("BareFnTy \\{fn_style: {:?}, abi: {}, sig: {}\\}",
                       self.fn_style,
                       self.abi.to_str(),
                       self.sig.repr(tcx))
    }
}

impl Repr for ty::FnSig {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        fn_sig_to_str(tcx, self)
    }
}

impl Repr for typeck::MethodCallee {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("MethodCallee \\{origin: {}, ty: {}, {}\\}",
                       self.origin.repr(tcx),
                       self.ty.repr(tcx),
                       self.substs.repr(tcx))
    }
}

impl Repr for typeck::MethodOrigin {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        match self {
            &typeck::MethodStatic(def_id) => {
                format_strbuf!("MethodStatic({})", def_id.repr(tcx))
            }
            &typeck::MethodParam(ref p) => {
                p.repr(tcx)
            }
            &typeck::MethodObject(ref p) => {
                p.repr(tcx)
            }
        }
    }
}

impl Repr for typeck::MethodParam {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("MethodParam({},{:?},{:?},{:?})",
                       self.trait_id.repr(tcx),
                       self.method_num,
                       self.param_num,
                       self.bound_num)
    }
}

impl Repr for typeck::MethodObject {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("MethodObject({},{:?},{:?})",
                       self.trait_id.repr(tcx),
                       self.method_num,
                       self.real_index)
    }
}


impl Repr for ty::RegionVid {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl Repr for ty::TraitStore {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        trait_store_to_str(tcx, *self)
    }
}

impl Repr for ty::BuiltinBound {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl UserString for ty::BuiltinBound {
    fn user_string(&self, _tcx: &ctxt) -> StrBuf {
        match *self {
            ty::BoundStatic => "'static".to_strbuf(),
            ty::BoundSend => "Send".to_strbuf(),
            ty::BoundSized => "Sized".to_strbuf(),
            ty::BoundCopy => "Copy".to_strbuf(),
            ty::BoundShare => "Share".to_strbuf(),
        }
    }
}

impl Repr for ty::BuiltinBounds {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        self.user_string(tcx)
    }
}

impl Repr for Span {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        tcx.sess.codemap().span_to_str(*self).to_strbuf()
    }
}

impl<A:UserString> UserString for Rc<A> {
    fn user_string(&self, tcx: &ctxt) -> StrBuf {
        let this: &A = &**self;
        this.user_string(tcx)
    }
}

impl UserString for ty::BuiltinBounds {
    fn user_string(&self, tcx: &ctxt) -> StrBuf {
        self.iter()
            .map(|bb| bb.user_string(tcx))
            .collect::<Vec<StrBuf>>()
            .connect("+")
            .to_strbuf()
    }
}

impl UserString for ty::TraitRef {
    fn user_string(&self, tcx: &ctxt) -> StrBuf {
        let base = ty::item_path_str(tcx, self.def_id);
        if tcx.sess.verbose() && self.substs.self_ty.is_some() {
            let mut all_tps = self.substs.tps.clone();
            for &t in self.substs.self_ty.iter() { all_tps.push(t); }
            parameterized(tcx,
                          base.as_slice(),
                          &self.substs.regions,
                          all_tps.as_slice(),
                          self.def_id,
                          true)
        } else {
            parameterized(tcx,
                          base.as_slice(),
                          &self.substs.regions,
                          self.substs.tps.as_slice(),
                          self.def_id,
                          true)
        }
    }
}

impl UserString for ty::t {
    fn user_string(&self, tcx: &ctxt) -> StrBuf {
        ty_to_str(tcx, *self)
    }
}

impl UserString for ast::Ident {
    fn user_string(&self, _tcx: &ctxt) -> StrBuf {
        token::get_name(self.name).get().to_strbuf()
    }
}

impl Repr for abi::Abi {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        self.to_str().to_strbuf()
    }
}

impl UserString for abi::Abi {
    fn user_string(&self, _tcx: &ctxt) -> StrBuf {
        self.to_str().to_strbuf()
    }
}

impl Repr for ty::UpvarId {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("UpvarId({};`{}`;{})",
                       self.var_id,
                       ty::local_var_name_str(tcx, self.var_id),
                       self.closure_expr_id)
    }
}

impl Repr for ast::Mutability {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl Repr for ty::BorrowKind {
    fn repr(&self, _tcx: &ctxt) -> StrBuf {
        format_strbuf!("{:?}", *self)
    }
}

impl Repr for ty::UpvarBorrow {
    fn repr(&self, tcx: &ctxt) -> StrBuf {
        format_strbuf!("UpvarBorrow({}, {})",
                       self.kind.repr(tcx),
                       self.region.repr(tcx))
    }
}
