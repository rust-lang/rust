// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::def;
use middle::subst::{VecPerParamSpace,Subst};
use middle::subst;
use middle::ty::{BoundRegion, BrAnon, BrNamed};
use middle::ty::{ReEarlyBound, BrFresh, ctxt};
use middle::ty::{ReFree, ReScope, ReInfer, ReStatic, Region, ReEmpty};
use middle::ty::{ReSkolemized, ReVar};
use middle::ty::{mt, t, ParamTy};
use middle::ty::{ty_bool, ty_char, ty_bot, ty_box, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_str, ty_vec, ty_float, ty_bare_fn, ty_closure};
use middle::ty::{ty_nil, ty_param, ty_ptr, ty_rptr, ty_tup, ty_open};
use middle::ty::{ty_unboxed_closure};
use middle::ty::{ty_uniq, ty_trait, ty_int, ty_uint, ty_infer};
use middle::ty;
use middle::typeck;
use middle::typeck::check::regionmanip;
use middle::typeck::infer;

use std::gc::Gc;
use std::rc::Rc;
use syntax::abi;
use syntax::ast_map;
use syntax::codemap::{Span, Pos};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::{ast, ast_util};
use syntax::owned_slice::OwnedSlice;

/// Produces a string suitable for debugging output.
pub trait Repr {
    fn repr(&self, tcx: &ctxt) -> String;
}

/// Produces a string suitable for showing to the user.
pub trait UserString {
    fn user_string(&self, tcx: &ctxt) -> String;
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

fn item_scope_tag(item: &ast::Item) -> &'static str {
    /*!
     * When a free region is associated with `item`, how should we describe
     * the item in the error message.
     */

    match item.node {
        ast::ItemImpl(..) => "impl",
        ast::ItemStruct(..) => "struct",
        ast::ItemEnum(..) => "enum",
        ast::ItemTrait(..) => "trait",
        ast::ItemFn(..) => "function body",
        _ => "item"
    }
}

pub fn explain_region_and_span(cx: &ctxt, region: ty::Region)
                            -> (String, Option<Span>) {
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
          Some(ast_map::NodeItem(it)) => {
              let tag = item_scope_tag(&*it);
              explain_span(cx, tag, it.span)
          }
          Some(_) | None => {
            // this really should not happen
            (format!("unknown scope: {}.  Please report a bug.", node_id), None)
          }
        }
      }

      ReFree(ref fr) => {
        let prefix = match fr.bound_region {
          BrAnon(idx) => {
              format!("the anonymous lifetime #{} defined on", idx + 1)
          }
          BrFresh(_) => "an anonymous lifetime defined on".to_string(),
          _ => {
              format!("the lifetime {} as defined on",
                      bound_region_ptr_to_string(cx, fr.bound_region))
          }
        };

        match cx.map.find(fr.scope_id) {
          Some(ast_map::NodeBlock(ref blk)) => {
              let (msg, opt_span) = explain_span(cx, "block", blk.span);
              (format!("{} {}", prefix, msg), opt_span)
          }
          Some(ast_map::NodeItem(it)) => {
              let tag = item_scope_tag(&*it);
              let (msg, opt_span) = explain_span(cx, tag, it.span);
              (format!("{} {}", prefix, msg), opt_span)
          }
          Some(_) | None => {
              // this really should not happen
              (format!("{} node {}", prefix, fr.scope_id), None)
          }
        }
      }

      ReStatic => { ("the static lifetime".to_string(), None) }

      ReEmpty => { ("the empty lifetime".to_string(), None) }

      ReEarlyBound(_, _, _, name) => {
        (format!("{}", token::get_name(name)), None)
      }

      // I believe these cases should not occur (except when debugging,
      // perhaps)
      ty::ReInfer(_) | ty::ReLateBound(..) => {
        (format!("lifetime {:?}", region), None)
      }
    };

    fn explain_span(cx: &ctxt, heading: &str, span: Span)
                    -> (String, Option<Span>) {
        let lo = cx.sess.codemap().lookup_char_pos_adj(span.lo);
        (format!("the {} at {}:{}", heading, lo.line, lo.col.to_uint()),
         Some(span))
    }
}

pub fn bound_region_ptr_to_string(cx: &ctxt, br: BoundRegion) -> String {
    bound_region_to_string(cx, "", false, br)
}

pub fn bound_region_to_string(cx: &ctxt,
                           prefix: &str, space: bool,
                           br: BoundRegion) -> String {
    let space_str = if space { " " } else { "" };

    if cx.sess.verbose() {
        return format!("{}{}{}", prefix, br.repr(cx), space_str)
    }

    match br {
        BrNamed(_, name) => {
            format!("{}{}{}", prefix, token::get_name(name), space_str)
        }
        BrAnon(_) => prefix.to_string(),
        BrFresh(_) => prefix.to_string(),
    }
}

// In general, if you are giving a region error message,
// you should use `explain_region()` or, better yet,
// `note_and_explain_region()`
pub fn region_ptr_to_string(cx: &ctxt, region: Region) -> String {
    region_to_string(cx, "&", true, region)
}

pub fn region_to_string(cx: &ctxt, prefix: &str, space: bool, region: Region) -> String {
    let space_str = if space { " " } else { "" };

    if cx.sess.verbose() {
        return format!("{}{}{}", prefix, region.repr(cx), space_str)
    }

    // These printouts are concise.  They do not contain all the information
    // the user might want to diagnose an error, but there is basically no way
    // to fit that into a short string.  Hence the recommendation to use
    // `explain_region()` or `note_and_explain_region()`.
    match region {
        ty::ReScope(_) => prefix.to_string(),
        ty::ReEarlyBound(_, _, _, name) => {
            token::get_name(name).get().to_string()
        }
        ty::ReLateBound(_, br) => bound_region_to_string(cx, prefix, space, br),
        ty::ReFree(ref fr) => bound_region_to_string(cx, prefix, space, fr.bound_region),
        ty::ReInfer(ReSkolemized(_, br)) => {
            bound_region_to_string(cx, prefix, space, br)
        }
        ty::ReInfer(ReVar(_)) => prefix.to_string(),
        ty::ReStatic => format!("{}'static{}", prefix, space_str),
        ty::ReEmpty => format!("{}'<empty>{}", prefix, space_str),
    }
}

pub fn mutability_to_string(m: ast::Mutability) -> String {
    match m {
        ast::MutMutable => "mut ".to_string(),
        ast::MutImmutable => "".to_string(),
    }
}

pub fn mt_to_string(cx: &ctxt, m: &mt) -> String {
    format!("{}{}", mutability_to_string(m.mutbl), ty_to_string(cx, m.ty))
}

pub fn trait_store_to_string(cx: &ctxt, s: ty::TraitStore) -> String {
    match s {
        ty::UniqTraitStore => "Box ".to_string(),
        ty::RegionTraitStore(r, m) => {
            format!("{}{}", region_ptr_to_string(cx, r), mutability_to_string(m))
        }
    }
}

pub fn vec_map_to_string<T>(ts: &[T], f: |t: &T| -> String) -> String {
    let tstrs = ts.iter().map(f).collect::<Vec<String>>();
    format!("[{}]", tstrs.connect(", "))
}

pub fn fn_sig_to_string(cx: &ctxt, typ: &ty::FnSig) -> String {
    format!("fn{}{} -> {}", typ.binder_id, typ.inputs.repr(cx),
            typ.output.repr(cx))
}

pub fn trait_ref_to_string(cx: &ctxt, trait_ref: &ty::TraitRef) -> String {
    trait_ref.user_string(cx).to_string()
}

pub fn ty_to_string(cx: &ctxt, typ: t) -> String {
    fn fn_input_to_string(cx: &ctxt, input: ty::t) -> String {
        ty_to_string(cx, input).to_string()
    }
    fn bare_fn_to_string(cx: &ctxt,
                      fn_style: ast::FnStyle,
                      abi: abi::Abi,
                      ident: Option<ast::Ident>,
                      sig: &ty::FnSig)
                      -> String {
        let mut s = String::new();
        match fn_style {
            ast::NormalFn => {}
            _ => {
                s.push_str(fn_style.to_string().as_slice());
                s.push_char(' ');
            }
        };

        if abi != abi::Rust {
            s.push_str(format!("extern {} ", abi.to_string()).as_slice());
        };

        s.push_str("fn");

        match ident {
            Some(i) => {
                s.push_char(' ');
                s.push_str(token::get_ident(i).get());
            }
            _ => { }
        }

        push_sig_to_string(cx, &mut s, '(', ')', sig, "");

        s
    }

    fn closure_to_string(cx: &ctxt, cty: &ty::ClosureTy) -> String {
        let mut s = String::new();

        match cty.store {
            ty::UniqTraitStore => {}
            ty::RegionTraitStore(region, _) => {
                s.push_str(region_to_string(cx, "", true, region).as_slice());
            }
        }

        match cty.fn_style {
            ast::NormalFn => {}
            _ => {
                s.push_str(cty.fn_style.to_string().as_slice());
                s.push_char(' ');
            }
        };

        let bounds_str = cty.bounds.user_string(cx);

        match cty.store {
            ty::UniqTraitStore => {
                assert_eq!(cty.onceness, ast::Once);
                s.push_str("proc");
                push_sig_to_string(cx, &mut s, '(', ')', &cty.sig,
                                   bounds_str.as_slice());
            }
            ty::RegionTraitStore(..) => {
                match cty.onceness {
                    ast::Many => {}
                    ast::Once => s.push_str("once ")
                }
                push_sig_to_string(cx, &mut s, '|', '|', &cty.sig,
                                   bounds_str.as_slice());
            }
        }

        s.into_owned()
    }

    fn push_sig_to_string(cx: &ctxt,
                       s: &mut String,
                       bra: char,
                       ket: char,
                       sig: &ty::FnSig,
                       bounds: &str) {
        s.push_char(bra);
        let strs: Vec<String> = sig.inputs.iter().map(|a| fn_input_to_string(cx, *a)).collect();
        s.push_str(strs.connect(", ").as_slice());
        if sig.variadic {
            s.push_str(", ...");
        }
        s.push_char(ket);

        if !bounds.is_empty() {
            s.push_str(":");
            s.push_str(bounds);
        }

        if ty::get(sig.output).sty != ty_nil {
            s.push_str(" -> ");
            if ty::type_is_bot(sig.output) {
                s.push_char('!');
            } else {
                s.push_str(ty_to_string(cx, sig.output).as_slice());
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
      ty_nil => "()".to_string(),
      ty_bot => "!".to_string(),
      ty_bool => "bool".to_string(),
      ty_char => "char".to_string(),
      ty_int(t) => ast_util::int_ty_to_string(t, None).to_string(),
      ty_uint(t) => ast_util::uint_ty_to_string(t, None).to_string(),
      ty_float(t) => ast_util::float_ty_to_string(t).to_string(),
      ty_box(typ) => format!("Gc<{}>", ty_to_string(cx, typ)),
      ty_uniq(typ) => format!("Box<{}>", ty_to_string(cx, typ)),
      ty_ptr(ref tm) => {
          format!("*{} {}", match tm.mutbl {
              ast::MutMutable => "mut",
              ast::MutImmutable => "const",
          }, ty_to_string(cx, tm.ty))
      }
      ty_rptr(r, ref tm) => {
          let mut buf = region_ptr_to_string(cx, r);
          buf.push_str(mt_to_string(cx, tm).as_slice());
          buf
      }
      ty_open(typ) => format!("opened<{}>", ty_to_string(cx, typ)),
      ty_tup(ref elems) => {
        let strs: Vec<String> = elems.iter().map(|elem| ty_to_string(cx, *elem)).collect();
        format!("({})", strs.connect(","))
      }
      ty_closure(ref f) => {
          closure_to_string(cx, &**f)
      }
      ty_bare_fn(ref f) => {
          bare_fn_to_string(cx, f.fn_style, f.abi, None, &f.sig)
      }
      ty_infer(infer_ty) => infer_ty.to_string(),
      ty_err => "[type error]".to_string(),
      ty_param(ref param_ty) => {
          param_ty.repr(cx)
      }
      ty_enum(did, ref substs) | ty_struct(did, ref substs) => {
          let base = ty::item_path_str(cx, did);
          let generics = ty::lookup_item_type(cx, did).generics;
          parameterized(cx, base.as_slice(), substs, &generics)
      }
      ty_trait(box ty::TyTrait {
          def_id: did, ref substs, ref bounds
      }) => {
          let base = ty::item_path_str(cx, did);
          let trait_def = ty::lookup_trait_def(cx, did);
          let ty = parameterized(cx, base.as_slice(),
                                 substs, &trait_def.generics);
          let bound_str = bounds.user_string(cx);
          let bound_sep = if bound_str.is_empty() { "" } else { "+" };
          format!("{}{}{}",
                  ty,
                  bound_sep,
                  bound_str)
      }
      ty_str => "str".to_string(),
      ty_unboxed_closure(..) => "closure".to_string(),
      ty_vec(t, sz) => {
          match sz {
              Some(n) => {
                  format!("[{}, .. {}]", ty_to_string(cx, t), n)
              }
              None => format!("[{}]", ty_to_string(cx, t)),
          }
      }
    }
}

pub fn explicit_self_category_to_str(category: &ty::ExplicitSelfCategory)
                                     -> &'static str {
    match *category {
        ty::StaticExplicitSelfCategory => "static",
        ty::ByValueExplicitSelfCategory => "self",
        ty::ByReferenceExplicitSelfCategory(_, ast::MutMutable) => {
            "&mut self"
        }
        ty::ByReferenceExplicitSelfCategory(_, ast::MutImmutable) => "&self",
        ty::ByBoxExplicitSelfCategory => "Box<self>",
    }
}

pub fn parameterized(cx: &ctxt,
                     base: &str,
                     substs: &subst::Substs,
                     generics: &ty::Generics)
                     -> String
{
    let mut strs = Vec::new();

    match substs.regions {
        subst::ErasedRegions => { }
        subst::NonerasedRegions(ref regions) => {
            for &r in regions.iter() {
                let s = region_to_string(cx, "", false, r);
                if !s.is_empty() {
                    strs.push(s)
                } else {
                    // This happens when the value of the region
                    // parameter is not easily serialized. This may be
                    // because the user omitted it in the first place,
                    // or because it refers to some block in the code,
                    // etc. I'm not sure how best to serialize this.
                    strs.push(format!("'_"));
                }
            }
        }
    }

    let tps = substs.types.get_slice(subst::TypeSpace);
    let ty_params = generics.types.get_slice(subst::TypeSpace);
    let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
    let num_defaults = if has_defaults && !cx.sess.verbose() {
        ty_params.iter().zip(tps.iter()).rev().take_while(|&(def, &actual)| {
            match def.default {
                Some(default) => default.subst(cx, substs) == actual,
                None => false
            }
        }).count()
    } else {
        0
    };

    for t in tps.slice_to(tps.len() - num_defaults).iter() {
        strs.push(ty_to_string(cx, *t))
    }

    if cx.sess.verbose() {
        for t in substs.types.get_slice(subst::SelfSpace).iter() {
            strs.push(format!("for {}", t.repr(cx)));
        }
    }

    if strs.len() > 0u {
        format!("{}<{}>", base, strs.connect(","))
    } else {
        format!("{}", base)
    }
}

pub fn ty_to_short_str(cx: &ctxt, typ: t) -> String {
    let mut s = typ.repr(cx).to_string();
    if s.len() >= 32u {
        s = s.as_slice().slice(0u, 32u).to_string();
    }
    return s;
}

impl<T:Repr> Repr for Option<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        match self {
            &None => "None".to_string(),
            &Some(ref t) => t.repr(tcx),
        }
    }
}

impl<T:Repr,U:Repr> Repr for Result<T,U> {
    fn repr(&self, tcx: &ctxt) -> String {
        match self {
            &Ok(ref t) => t.repr(tcx),
            &Err(ref u) => format!("Err({})", u.repr(tcx))
        }
    }
}

impl Repr for () {
    fn repr(&self, _tcx: &ctxt) -> String {
        "()".to_string()
    }
}

impl<T:Repr> Repr for Rc<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        (&**self).repr(tcx)
    }
}

impl<T:Repr + 'static> Repr for Gc<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        (&**self).repr(tcx)
    }
}

impl<T:Repr> Repr for Box<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        (&**self).repr(tcx)
    }
}

fn repr_vec<T:Repr>(tcx: &ctxt, v: &[T]) -> String {
    vec_map_to_string(v, |t| t.repr(tcx))
}

impl<'a, T:Repr> Repr for &'a [T] {
    fn repr(&self, tcx: &ctxt) -> String {
        repr_vec(tcx, *self)
    }
}

impl<T:Repr> Repr for OwnedSlice<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        repr_vec(tcx, self.as_slice())
    }
}

// This is necessary to handle types like Option<~[T]>, for which
// autoderef cannot convert the &[T] handler
impl<T:Repr> Repr for Vec<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        repr_vec(tcx, self.as_slice())
    }
}

impl<T:UserString> UserString for Vec<T> {
    fn user_string(&self, tcx: &ctxt) -> String {
        let strs: Vec<String> =
            self.iter().map(|t| t.user_string(tcx)).collect();
        strs.connect(", ")
    }
}

impl Repr for def::Def {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::TypeParameterDef {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("TypeParameterDef({}, {})",
                self.def_id.repr(tcx),
                self.bounds.repr(tcx))
    }
}

impl Repr for ty::RegionParameterDef {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("RegionParameterDef(name={}, def_id={}, bounds={})",
                token::get_name(self.name),
                self.def_id.repr(tcx),
                self.bounds.repr(tcx))
    }
}

impl Repr for ty::t {
    fn repr(&self, tcx: &ctxt) -> String {
        ty_to_string(tcx, *self)
    }
}

impl Repr for ty::mt {
    fn repr(&self, tcx: &ctxt) -> String {
        mt_to_string(tcx, self)
    }
}

impl Repr for subst::Substs {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("Substs[types={}, regions={}]",
                       self.types.repr(tcx),
                       self.regions.repr(tcx))
    }
}

impl<T:Repr> Repr for subst::VecPerParamSpace<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("[{};{};{}]",
                       self.get_slice(subst::TypeSpace).repr(tcx),
                       self.get_slice(subst::SelfSpace).repr(tcx),
                       self.get_slice(subst::FnSpace).repr(tcx))
    }
}

impl Repr for ty::ItemSubsts {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ItemSubsts({})", self.substs.repr(tcx))
    }
}

impl Repr for subst::RegionSubsts {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            subst::ErasedRegions => "erased".to_string(),
            subst::NonerasedRegions(ref regions) => regions.repr(tcx)
        }
    }
}

impl Repr for ty::BuiltinBounds {
    fn repr(&self, _tcx: &ctxt) -> String {
        let mut res = Vec::new();
        for b in self.iter() {
            res.push(match b {
                ty::BoundSend => "Send".to_owned(),
                ty::BoundSized => "Sized".to_owned(),
                ty::BoundCopy => "Copy".to_owned(),
                ty::BoundSync => "Sync".to_owned(),
            });
        }
        res.connect("+")
    }
}

impl Repr for ty::ExistentialBounds {
    fn repr(&self, tcx: &ctxt) -> String {
        self.user_string(tcx)
    }
}

impl Repr for ty::ParamBounds {
    fn repr(&self, tcx: &ctxt) -> String {
        let mut res = Vec::new();
        res.push(self.builtin_bounds.repr(tcx));
        for t in self.trait_bounds.iter() {
            res.push(t.repr(tcx));
        }
        res.connect("+")
    }
}

impl Repr for ty::TraitRef {
    fn repr(&self, tcx: &ctxt) -> String {
        trait_ref_to_string(tcx, self)
    }
}

impl Repr for ty::TraitDef {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("TraitDef(generics={}, bounds={}, trait_ref={})",
                self.generics.repr(tcx),
                self.bounds.repr(tcx),
                self.trait_ref.repr(tcx))
    }
}

impl Repr for ast::Expr {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

impl Repr for ast::Path {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("path({})", pprust::path_to_string(self))
    }
}

impl UserString for ast::Path {
    fn user_string(&self, _tcx: &ctxt) -> String {
        pprust::path_to_string(self)
    }
}

impl Repr for ast::Item {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("item({})", tcx.map.node_to_string(self.id))
    }
}

impl Repr for ast::Lifetime {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

impl Repr for ast::Stmt {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("stmt({}: {})",
                ast_util::stmt_id(self),
                pprust::stmt_to_string(self))
    }
}

impl Repr for ast::Pat {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("pat({}: {})", self.id, pprust::pat_to_string(self))
    }
}

impl Repr for ty::BoundRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::BrAnon(id) => format!("BrAnon({})", id),
            ty::BrNamed(id, name) => {
                format!("BrNamed({}, {})", id.repr(tcx), token::get_name(name))
            }
            ty::BrFresh(id) => format!("BrFresh({})", id),
        }
    }
}

impl Repr for ty::Region {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::ReEarlyBound(id, space, index, name) => {
                format!("ReEarlyBound({}, {}, {}, {})",
                               id,
                               space,
                               index,
                               token::get_name(name))
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                format!("ReLateBound({}, {})",
                        binder_id,
                        bound_region.repr(tcx))
            }

            ty::ReFree(ref fr) => fr.repr(tcx),

            ty::ReScope(id) => {
                format!("ReScope({})", id)
            }

            ty::ReStatic => {
                "ReStatic".to_string()
            }

            ty::ReInfer(ReVar(ref vid)) => {
                format!("ReInfer({})", vid.index)
            }

            ty::ReInfer(ReSkolemized(id, ref bound_region)) => {
                format!("re_skolemized({}, {})", id, bound_region.repr(tcx))
            }

            ty::ReEmpty => {
                "ReEmpty".to_string()
            }
        }
    }
}

impl UserString for ty::Region {
    fn user_string(&self, tcx: &ctxt) -> String {
        region_to_string(tcx, "", false, *self)
    }
}

impl Repr for ty::FreeRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ReFree({}, {})",
                self.scope_id,
                self.bound_region.repr(tcx))
    }
}

impl Repr for ast::DefId {
    fn repr(&self, tcx: &ctxt) -> String {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.krate == ast::LOCAL_CRATE {
            {
                match tcx.map.find(self.node) {
                    Some(ast_map::NodeItem(..)) |
                    Some(ast_map::NodeForeignItem(..)) |
                    Some(ast_map::NodeImplItem(..)) |
                    Some(ast_map::NodeTraitItem(..)) |
                    Some(ast_map::NodeVariant(..)) |
                    Some(ast_map::NodeStructCtor(..)) => {
                        return format!(
                                "{:?}:{}",
                                *self,
                                ty::item_path_str(tcx, *self))
                    }
                    _ => {}
                }
            }
        }
        return format!("{:?}", *self)
    }
}

impl Repr for ty::Polytype {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("Polytype {{generics: {}, ty: {}}}",
                self.generics.repr(tcx),
                self.ty.repr(tcx))
    }
}

impl Repr for ty::Generics {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("Generics(types: {}, regions: {})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl Repr for ty::ItemVariances {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ItemVariances(types={}, \
                regions={})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl Repr for ty::Variance {
    fn repr(&self, _: &ctxt) -> String {
        // The first `.to_string()` returns a &'static str (it is not an implementation
        // of the ToString trait). Because of that, we need to call `.to_string()` again
        // if we want to have a `String`.
        self.to_string().to_string()
    }
}

impl Repr for ty::Method {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("method(ident: {}, generics: {}, fty: {}, \
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
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).get().to_string()
    }
}

impl UserString for ast::Name {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).get().to_string()
    }
}

impl Repr for ast::Ident {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_ident(*self).get().to_string()
    }
}

impl Repr for ast::ExplicitSelf_ {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::Visibility {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::BareFnTy {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("BareFnTy {{fn_style: {:?}, abi: {}, sig: {}}}",
                self.fn_style,
                self.abi.to_string(),
                self.sig.repr(tcx))
    }
}

impl Repr for ty::FnSig {
    fn repr(&self, tcx: &ctxt) -> String {
        fn_sig_to_string(tcx, self)
    }
}

impl Repr for typeck::MethodCallee {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("MethodCallee {{origin: {}, ty: {}, {}}}",
                self.origin.repr(tcx),
                self.ty.repr(tcx),
                self.substs.repr(tcx))
    }
}

impl Repr for typeck::MethodOrigin {
    fn repr(&self, tcx: &ctxt) -> String {
        match self {
            &typeck::MethodStatic(def_id) => {
                format!("MethodStatic({})", def_id.repr(tcx))
            }
            &typeck::MethodStaticUnboxedClosure(def_id) => {
                format!("MethodStaticUnboxedClosure({})", def_id.repr(tcx))
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
    fn repr(&self, tcx: &ctxt) -> String {
        format!("MethodParam({},{:?},{:?},{:?})",
                self.trait_id.repr(tcx),
                self.method_num,
                self.param_num,
                self.bound_num)
    }
}

impl Repr for typeck::MethodObject {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("MethodObject({},{:?},{:?})",
                self.trait_id.repr(tcx),
                self.method_num,
                self.real_index)
    }
}

impl Repr for ty::TraitStore {
    fn repr(&self, tcx: &ctxt) -> String {
        trait_store_to_string(tcx, *self)
    }
}

impl Repr for ty::BuiltinBound {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl UserString for ty::BuiltinBound {
    fn user_string(&self, _tcx: &ctxt) -> String {
        match *self {
            ty::BoundSend => "Send".to_owned(),
            ty::BoundSized => "Sized".to_owned(),
            ty::BoundCopy => "Copy".to_owned(),
            ty::BoundSync => "Sync".to_owned(),
        }
    }
}

impl Repr for Span {
    fn repr(&self, tcx: &ctxt) -> String {
        tcx.sess.codemap().span_to_string(*self).to_string()
    }
}

impl<A:UserString> UserString for Rc<A> {
    fn user_string(&self, tcx: &ctxt) -> String {
        let this: &A = &**self;
        this.user_string(tcx)
    }
}

impl UserString for ty::ParamBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        let mut result = Vec::new();
        let s = self.builtin_bounds.user_string(tcx);
        if !s.is_empty() {
            result.push(s);
        }
        for n in self.trait_bounds.iter() {
            result.push(n.user_string(tcx));
        }
        result.connect("+")
    }
}

impl UserString for ty::ExistentialBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        if self.builtin_bounds.contains_elem(ty::BoundSend) &&
            self.region_bound == ty::ReStatic
        { // Region bound is implied by builtin bounds:
            return self.builtin_bounds.repr(tcx);
        }

        let mut res = Vec::new();

        let region_str = self.region_bound.user_string(tcx);
        if !region_str.is_empty() {
            res.push(region_str);
        }

        for bound in self.builtin_bounds.iter() {
            res.push(bound.user_string(tcx));
        }

        res.connect("+")
    }
}

impl UserString for ty::BuiltinBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        self.iter()
            .map(|bb| bb.user_string(tcx))
            .collect::<Vec<String>>()
            .connect("+")
            .to_string()
    }
}

impl UserString for ty::TraitRef {
    fn user_string(&self, tcx: &ctxt) -> String {
        let base = ty::item_path_str(tcx, self.def_id);
        let trait_def = ty::lookup_trait_def(tcx, self.def_id);
        parameterized(tcx, base.as_slice(), &self.substs, &trait_def.generics)
    }
}

impl UserString for ty::t {
    fn user_string(&self, tcx: &ctxt) -> String {
        ty_to_string(tcx, *self)
    }
}

impl UserString for ast::Ident {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(self.name).get().to_string()
    }
}

impl Repr for abi::Abi {
    fn repr(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl UserString for abi::Abi {
    fn user_string(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl Repr for ty::UpvarId {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarId({};`{}`;{})",
                self.var_id,
                ty::local_var_name_str(tcx, self.var_id),
                self.closure_expr_id)
    }
}

impl Repr for ast::Mutability {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::BorrowKind {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::UpvarBorrow {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarBorrow({}, {})",
                self.kind.repr(tcx),
                self.region.repr(tcx))
    }
}

impl Repr for ty::IntVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl Repr for ty::FloatVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl Repr for ty::RegionVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl Repr for ty::TyVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl Repr for ty::IntVarValue {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::IntTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::UintTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::FloatTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<T:Repr> Repr for infer::Bounds<T> {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("({} <= {})",
                self.lb.repr(tcx),
                self.ub.repr(tcx))
    }
}

impl Repr for ty::ExplicitSelfCategory {
    fn repr(&self, _: &ctxt) -> String {
        explicit_self_category_to_str(self).to_string()
    }
}


impl Repr for regionmanip::WfConstraint {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            regionmanip::RegionSubRegionConstraint(_, r_a, r_b) => {
                format!("RegionSubRegionConstraint({}, {})",
                        r_a.repr(tcx),
                        r_b.repr(tcx))
            }

            regionmanip::RegionSubParamConstraint(_, r, p) => {
                format!("RegionSubParamConstraint({}, {})",
                        r.repr(tcx),
                        p.repr(tcx))
            }
        }
    }
}

impl UserString for ParamTy {
    fn user_string(&self, tcx: &ctxt) -> String {
        let id = self.idx;
        let did = self.def_id;
        let ident = match tcx.ty_param_defs.borrow().find(&did.node) {
            Some(def) => token::get_ident(def.ident).get().to_string(),

            // This can only happen when a type mismatch error happens and
            // the actual type has more type parameters than the expected one.
            None => format!("<generic #{}>", id),
        };
        ident
    }
}

impl Repr for ParamTy {
    fn repr(&self, tcx: &ctxt) -> String {
        self.user_string(tcx)
    }
}

impl<A:Repr,B:Repr> Repr for (A,B) {
    fn repr(&self, tcx: &ctxt) -> String {
        let &(ref a, ref b) = self;
        format!("({},{})", a.repr(tcx), b.repr(tcx))
    }
}
