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
use middle::ty::{ReSkolemized, ReVar, BrEnv};
use middle::ty::{mt, Ty, ParamTy};
use middle::ty::{ty_bool, ty_char, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_str, ty_vec, ty_float, ty_bare_fn, ty_closure};
use middle::ty::{ty_param, ty_ptr, ty_rptr, ty_tup, ty_open};
use middle::ty::{ty_unboxed_closure};
use middle::ty::{ty_uniq, ty_trait, ty_int, ty_uint, ty_infer};
use middle::ty;
use middle::typeck;
use middle::typeck::check::regionmanip;

use std::rc::Rc;
use syntax::abi;
use syntax::ast_map;
use syntax::codemap::{Span, Pos};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::{ast, ast_util};
use syntax::owned_slice::OwnedSlice;

/// Produces a string suitable for debugging output.
pub trait Repr<'tcx> for Sized? {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String;
}

/// Produces a string suitable for showing to the user.
pub trait UserString<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String;
}

pub fn note_and_explain_region(cx: &ctxt,
                               prefix: &str,
                               region: ty::Region,
                               suffix: &str) -> Option<Span> {
    match explain_region_and_span(cx, region) {
      (ref str, Some(span)) => {
        cx.sess.span_note(
            span,
            format!("{}{}{}", prefix, *str, suffix).as_slice());
        Some(span)
      }
      (ref str, None) => {
        cx.sess.note(
            format!("{}{}{}", prefix, *str, suffix).as_slice());
        None
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
      ReScope(scope) => {
        match cx.map.find(scope.node_id()) {
          Some(ast_map::NodeBlock(ref blk)) => {
            explain_span(cx, "block", blk.span)
          }
          Some(ast_map::NodeExpr(expr)) => {
            match expr.node {
              ast::ExprCall(..) => explain_span(cx, "call", expr.span),
              ast::ExprMethodCall(..) => {
                explain_span(cx, "method call", expr.span)
              },
              ast::ExprMatch(_, _, ast::MatchIfLetDesugar) => explain_span(cx, "if let", expr.span),
              ast::ExprMatch(_, _, ast::MatchWhileLetDesugar) => {
                  explain_span(cx, "while let", expr.span)
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
            (format!("unknown scope: {}.  Please report a bug.", scope), None)
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

        match cx.map.find(fr.scope.node_id()) {
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
              (format!("{} unknown free region bounded by scope {}", prefix, fr.scope), None)
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
        (format!("lifetime {}", region), None)
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
        BrAnon(_) | BrFresh(_) | BrEnv => prefix.to_string()
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

pub fn mt_to_string<'tcx>(cx: &ctxt<'tcx>, m: &mt<'tcx>) -> String {
    format!("{}{}",
        mutability_to_string(m.mutbl),
        ty_to_string(cx, m.ty))
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

pub fn fn_sig_to_string<'tcx>(cx: &ctxt<'tcx>, typ: &ty::FnSig<'tcx>) -> String {
    format!("fn{} -> {}", typ.inputs.repr(cx), typ.output.repr(cx))
}

pub fn trait_ref_to_string<'tcx>(cx: &ctxt<'tcx>,
                                 trait_ref: &ty::TraitRef<'tcx>) -> String {
    trait_ref.user_string(cx).to_string()
}

pub fn ty_to_string<'tcx>(cx: &ctxt<'tcx>, typ: &ty::TyS<'tcx>) -> String {
    fn bare_fn_to_string<'tcx>(cx: &ctxt<'tcx>,
                               fn_style: ast::FnStyle,
                               abi: abi::Abi,
                               ident: Option<ast::Ident>,
                               sig: &ty::FnSig<'tcx>)
                               -> String {
        let mut s = String::new();
        match fn_style {
            ast::NormalFn => {}
            _ => {
                s.push_str(fn_style.to_string().as_slice());
                s.push(' ');
            }
        };

        if abi != abi::Rust {
            s.push_str(format!("extern {} ", abi.to_string()).as_slice());
        };

        s.push_str("fn");

        match ident {
            Some(i) => {
                s.push(' ');
                s.push_str(token::get_ident(i).get());
            }
            _ => { }
        }

        push_sig_to_string(cx, &mut s, '(', ')', sig, "");

        s
    }

    fn closure_to_string<'tcx>(cx: &ctxt<'tcx>, cty: &ty::ClosureTy<'tcx>) -> String {
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
                s.push(' ');
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

        s
    }

    fn push_sig_to_string<'tcx>(cx: &ctxt<'tcx>,
                                s: &mut String,
                                bra: char,
                                ket: char,
                                sig: &ty::FnSig<'tcx>,
                                bounds: &str) {
        s.push(bra);
        let strs = sig.inputs
            .iter()
            .map(|a| ty_to_string(cx, *a))
            .collect::<Vec<_>>();
        s.push_str(strs.connect(", ").as_slice());
        if sig.variadic {
            s.push_str(", ...");
        }
        s.push(ket);

        if !bounds.is_empty() {
            s.push_str(":");
            s.push_str(bounds);
        }

        match sig.output {
            ty::FnConverging(t) => {
                if !ty::type_is_nil(t) {
                   s.push_str(" -> ");
                   s.push_str(ty_to_string(cx, t).as_slice());
                }
            }
            ty::FnDiverging => {
                s.push_str(" -> !");
            }
        }
    }

    fn infer_ty_to_string(cx: &ctxt, ty: ty::InferTy) -> String {
        let print_var_ids = cx.sess.verbose();
        match ty {
            ty::TyVar(ref vid) if print_var_ids => vid.repr(cx),
            ty::IntVar(ref vid) if print_var_ids => vid.repr(cx),
            ty::FloatVar(ref vid) if print_var_ids => vid.repr(cx),
            ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => format!("_"),
            ty::SkolemizedTy(v) => format!("SkolemizedTy({})", v),
            ty::SkolemizedIntTy(v) => format!("SkolemizedIntTy({})", v)
        }
    }

    // pretty print the structural type representation:
    match typ.sty {
        ty_bool => "bool".to_string(),
        ty_char => "char".to_string(),
        ty_int(t) => ast_util::int_ty_to_string(t, None).to_string(),
        ty_uint(t) => ast_util::uint_ty_to_string(t, None).to_string(),
        ty_float(t) => ast_util::float_ty_to_string(t).to_string(),
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
        ty_open(typ) =>
            format!("opened<{}>", ty_to_string(cx, typ)),
        ty_tup(ref elems) => {
            let strs = elems
                .iter()
                .map(|elem| ty_to_string(cx, *elem))
                .collect::<Vec<_>>();
            match strs.as_slice() {
                [ref string] => format!("({},)", string),
                strs => format!("({})", strs.connect(", "))
            }
        }
        ty_closure(ref f) => {
            closure_to_string(cx, &**f)
        }
        ty_bare_fn(ref f) => {
            bare_fn_to_string(cx, f.fn_style, f.abi, None, &f.sig)
        }
        ty_infer(infer_ty) => infer_ty_to_string(cx, infer_ty),
        ty_err => "[type error]".to_string(),
        ty_param(ref param_ty) => {
            if cx.sess.verbose() {
                param_ty.repr(cx)
            } else {
                param_ty.user_string(cx)
            }
        }
        ty_enum(did, ref substs) | ty_struct(did, ref substs) => {
            let base = ty::item_path_str(cx, did);
            let generics = ty::lookup_item_type(cx, did).generics;
            parameterized(cx, base.as_slice(), substs, &generics)
        }
        ty_trait(box ty::TyTrait {
            ref principal, ref bounds
        }) => {
            let base = ty::item_path_str(cx, principal.def_id);
            let trait_def = ty::lookup_trait_def(cx, principal.def_id);
            let ty = parameterized(cx, base.as_slice(),
                                   &principal.substs, &trait_def.generics);
            let bound_str = bounds.user_string(cx);
            let bound_sep = if bound_str.is_empty() { "" } else { "+" };
            format!("{}{}{}",
                    ty,
                    bound_sep,
                    bound_str)
        }
        ty_str => "str".to_string(),
        ty_unboxed_closure(ref did, _, ref substs) => {
            let unboxed_closures = cx.unboxed_closures.borrow();
            unboxed_closures.get(did).map(|cl| {
                closure_to_string(cx, &cl.closure_type.subst(cx, substs))
            }).unwrap_or_else(|| "closure".to_string())
        }
        ty_vec(t, sz) => {
            let inner_str = ty_to_string(cx, t);
            match sz {
                Some(n) => format!("[{}, ..{}]", inner_str, n),
                None => format!("[{}]", inner_str),
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

pub fn parameterized<'tcx>(cx: &ctxt<'tcx>,
                           base: &str,
                           substs: &subst::Substs<'tcx>,
                           generics: &ty::Generics<'tcx>)
                           -> String
{
    if cx.sess.verbose() {
        if substs.is_noop() {
            return format!("{}", base);
        } else {
            return format!("{}<{},{}>",
                           base,
                           substs.regions.repr(cx),
                           substs.types.repr(cx));
        }
    }

    let mut strs = Vec::new();

    match substs.regions {
        subst::ErasedRegions => { }
        subst::NonerasedRegions(ref regions) => {
            for &r in regions.iter() {
                let s = region_to_string(cx, "", false, r);
                if s.is_empty() {
                    // This happens when the value of the region
                    // parameter is not easily serialized. This may be
                    // because the user omitted it in the first place,
                    // or because it refers to some block in the code,
                    // etc. I'm not sure how best to serialize this.
                    strs.push(format!("'_"));
                } else {
                    strs.push(s)
                }
            }
        }
    }

    let tps = substs.types.get_slice(subst::TypeSpace);
    let ty_params = generics.types.get_slice(subst::TypeSpace);
    let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
    let num_defaults = if has_defaults {
        ty_params.iter().zip(tps.iter()).rev().take_while(|&(def, &actual)| {
            match def.default {
                Some(default) => default.subst(cx, substs) == actual,
                None => false
            }
        }).count()
    } else {
        0
    };

    for t in tps[..tps.len() - num_defaults].iter() {
        strs.push(ty_to_string(cx, *t))
    }

    if strs.len() > 0u {
        format!("{}<{}>", base, strs.connect(", "))
    } else {
        format!("{}", base)
    }
}

pub fn ty_to_short_str<'tcx>(cx: &ctxt<'tcx>, typ: Ty<'tcx>) -> String {
    let mut s = typ.repr(cx).to_string();
    if s.len() >= 32u {
        s = s.as_slice().slice(0u, 32u).to_string();
    }
    return s;
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Option<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &None => "None".to_string(),
            &Some(ref t) => t.repr(tcx),
        }
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for P<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (*self).repr(tcx)
    }
}

impl<'tcx,T:Repr<'tcx>,U:Repr<'tcx>> Repr<'tcx> for Result<T,U> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &Ok(ref t) => t.repr(tcx),
            &Err(ref u) => format!("Err({})", u.repr(tcx))
        }
    }
}

impl<'tcx> Repr<'tcx> for () {
    fn repr(&self, _tcx: &ctxt) -> String {
        "()".to_string()
    }
}

impl<'a, 'tcx, Sized? T:Repr<'tcx>> Repr<'tcx> for &'a T {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        Repr::repr(*self, tcx)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Rc<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (&**self).repr(tcx)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Box<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (&**self).repr(tcx)
    }
}

fn repr_vec<'tcx, T:Repr<'tcx>>(tcx: &ctxt<'tcx>, v: &[T]) -> String {
    vec_map_to_string(v, |t| t.repr(tcx))
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for [T] {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        repr_vec(tcx, self)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for OwnedSlice<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        repr_vec(tcx, self.as_slice())
    }
}

// This is necessary to handle types like Option<~[T]>, for which
// autoderef cannot convert the &[T] handler
impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Vec<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        repr_vec(tcx, self.as_slice())
    }
}

impl<'tcx, T:UserString<'tcx>> UserString<'tcx> for Vec<T> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let strs: Vec<String> =
            self.iter().map(|t| t.user_string(tcx)).collect();
        strs.connect(", ")
    }
}

impl<'tcx> Repr<'tcx> for def::Def {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TypeParameterDef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TypeParameterDef({}, {}, {}/{})",
                self.def_id,
                self.bounds.repr(tcx),
                self.space,
                self.index)
    }
}

impl<'tcx> Repr<'tcx> for ty::RegionParameterDef {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("RegionParameterDef(name={}, def_id={}, bounds={})",
                token::get_name(self.name),
                self.def_id.repr(tcx),
                self.bounds.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::TyS<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        ty_to_string(tcx, self)
    }
}

impl<'tcx> Repr<'tcx> for ty::mt<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        mt_to_string(tcx, self)
    }
}

impl<'tcx> Repr<'tcx> for subst::Substs<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Substs[types={}, regions={}]",
                       self.types.repr(tcx),
                       self.regions.repr(tcx))
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for subst::VecPerParamSpace<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("[{};{};{};{}]",
                self.get_slice(subst::TypeSpace).repr(tcx),
                self.get_slice(subst::SelfSpace).repr(tcx),
                self.get_slice(subst::AssocSpace).repr(tcx),
                self.get_slice(subst::FnSpace).repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ItemSubsts<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ItemSubsts({})", self.substs.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for subst::RegionSubsts {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            subst::ErasedRegions => "erased".to_string(),
            subst::NonerasedRegions(ref regions) => regions.repr(tcx)
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::BuiltinBounds {
    fn repr(&self, _tcx: &ctxt) -> String {
        let mut res = Vec::new();
        for b in self.iter() {
            res.push(match b {
                ty::BoundSend => "Send".to_string(),
                ty::BoundSized => "Sized".to_string(),
                ty::BoundCopy => "Copy".to_string(),
                ty::BoundSync => "Sync".to_string(),
            });
        }
        res.connect("+")
    }
}

impl<'tcx> Repr<'tcx> for ty::ExistentialBounds {
    fn repr(&self, tcx: &ctxt) -> String {
        self.user_string(tcx)
    }
}

impl<'tcx> Repr<'tcx> for ty::ParamBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let mut res = Vec::new();
        res.push(self.builtin_bounds.repr(tcx));
        for t in self.trait_bounds.iter() {
            res.push(t.repr(tcx));
        }
        res.connect("+")
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitRef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        // when printing out the debug representation, we don't need
        // to enumerate the `for<...>` etc because the debruijn index
        // tells you everything you need to know.
        let base = ty::item_path_str(tcx, self.def_id);
        let trait_def = ty::lookup_trait_def(tcx, self.def_id);
        format!("<{} : {}>",
                self.substs.self_ty().repr(tcx),
                parameterized(tcx, base.as_slice(), &self.substs, &trait_def.generics))
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitDef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TraitDef(generics={}, bounds={}, trait_ref={})",
                self.generics.repr(tcx),
                self.bounds.repr(tcx),
                self.trait_ref.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ast::TraitItem {
    fn repr(&self, _tcx: &ctxt) -> String {
        match *self {
            ast::RequiredMethod(ref data) => format!("RequiredMethod({}, id={})",
                                                     data.ident, data.id),
            ast::ProvidedMethod(ref data) => format!("ProvidedMethod(id={})",
                                                     data.id),
            ast::TypeTraitItem(ref data) => format!("TypeTraitItem({}, id={})",
                                                     data.ty_param.ident, data.ty_param.id),
        }
    }
}

impl<'tcx> Repr<'tcx> for ast::Expr {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Path {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("path({})", pprust::path_to_string(self))
    }
}

impl<'tcx> UserString<'tcx> for ast::Path {
    fn user_string(&self, _tcx: &ctxt) -> String {
        pprust::path_to_string(self)
    }
}

impl<'tcx> Repr<'tcx> for ast::Ty {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("type({})", pprust::ty_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Item {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("item({})", tcx.map.node_to_string(self.id))
    }
}

impl<'tcx> Repr<'tcx> for ast::Lifetime {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Stmt {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("stmt({}: {})",
                ast_util::stmt_id(self),
                pprust::stmt_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Pat {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("pat({}: {})", self.id, pprust::pat_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ty::BoundRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::BrAnon(id) => format!("BrAnon({})", id),
            ty::BrNamed(id, name) => {
                format!("BrNamed({}, {})", id.repr(tcx), token::get_name(name))
            }
            ty::BrFresh(id) => format!("BrFresh({})", id),
            ty::BrEnv => "BrEnv".to_string()
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::Region {
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
                format!("{}", vid)
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

impl<'tcx> UserString<'tcx> for ty::Region {
    fn user_string(&self, tcx: &ctxt) -> String {
        region_to_string(tcx, "", false, *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::FreeRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ReFree({}, {})",
                self.scope.node_id(),
                self.bound_region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ast::DefId {
    fn repr(&self, tcx: &ctxt) -> String {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.krate == ast::LOCAL_CRATE {
            match tcx.map.find(self.node) {
                Some(ast_map::NodeItem(..)) |
                Some(ast_map::NodeForeignItem(..)) |
                Some(ast_map::NodeImplItem(..)) |
                Some(ast_map::NodeTraitItem(..)) |
                Some(ast_map::NodeVariant(..)) |
                Some(ast_map::NodeStructCtor(..)) => {
                    return format!(
                                "{}:{}",
                                *self,
                                ty::item_path_str(tcx, *self))
                }
                _ => {}
            }
        }
        return format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::Polytype<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Polytype {{generics: {}, ty: {}}}",
                self.generics.repr(tcx),
                self.ty.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Generics<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Generics(types: {}, regions: {})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::GenericBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("GenericBounds(types: {}, regions: {})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ItemVariances {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ItemVariances(types={}, \
                regions={})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Variance {
    fn repr(&self, _: &ctxt) -> String {
        // The first `.to_string()` returns a &'static str (it is not an implementation
        // of the ToString trait). Because of that, we need to call `.to_string()` again
        // if we want to have a `String`.
        let result: &'static str = (*self).to_string();
        result.to_string()
    }
}

impl<'tcx> Repr<'tcx> for ty::Method<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("method(name: {}, generics: {}, fty: {}, \
                 explicit_self: {}, vis: {}, def_id: {})",
                self.name.repr(tcx),
                self.generics.repr(tcx),
                self.fty.repr(tcx),
                self.explicit_self.repr(tcx),
                self.vis.repr(tcx),
                self.def_id.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ast::Name {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).get().to_string()
    }
}

impl<'tcx> UserString<'tcx> for ast::Name {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).get().to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::Ident {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_ident(*self).get().to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::ExplicitSelf_ {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::Visibility {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BareFnTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("BareFnTy {{fn_style: {}, abi: {}, sig: {}}}",
                self.fn_style,
                self.abi.to_string(),
                self.sig.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::FnSig<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        fn_sig_to_string(tcx, self)
    }
}

impl<'tcx> Repr<'tcx> for ty::FnOutput<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            ty::FnConverging(ty) =>
                format!("FnConverging({0})", ty.repr(tcx)),
            ty::FnDiverging =>
                "FnDiverging".to_string()
        }
    }
}

impl<'tcx> Repr<'tcx> for typeck::MethodCallee<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodCallee {{origin: {}, ty: {}, {}}}",
                self.origin.repr(tcx),
                self.ty.repr(tcx),
                self.substs.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for typeck::MethodOrigin<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &typeck::MethodStatic(def_id) => {
                format!("MethodStatic({})", def_id.repr(tcx))
            }
            &typeck::MethodStaticUnboxedClosure(def_id) => {
                format!("MethodStaticUnboxedClosure({})", def_id.repr(tcx))
            }
            &typeck::MethodTypeParam(ref p) => {
                p.repr(tcx)
            }
            &typeck::MethodTraitObject(ref p) => {
                p.repr(tcx)
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for typeck::MethodParam<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodParam({},{})",
                self.trait_ref.repr(tcx),
                self.method_num)
    }
}

impl<'tcx> Repr<'tcx> for typeck::MethodObject<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodObject({},{},{})",
                self.trait_ref.repr(tcx),
                self.method_num,
                self.real_index)
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitStore {
    fn repr(&self, tcx: &ctxt) -> String {
        trait_store_to_string(tcx, *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BuiltinBound {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> UserString<'tcx> for ty::BuiltinBound {
    fn user_string(&self, _tcx: &ctxt) -> String {
        match *self {
            ty::BoundSend => "Send".to_string(),
            ty::BoundSized => "Sized".to_string(),
            ty::BoundCopy => "Copy".to_string(),
            ty::BoundSync => "Sync".to_string(),
        }
    }
}

impl<'tcx> Repr<'tcx> for Span {
    fn repr(&self, tcx: &ctxt) -> String {
        tcx.sess.codemap().span_to_string(*self).to_string()
    }
}

impl<'tcx, A:UserString<'tcx>> UserString<'tcx> for Rc<A> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let this: &A = &**self;
        this.user_string(tcx)
    }
}

impl<'tcx> UserString<'tcx> for ty::ParamBounds<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
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

impl<'tcx> UserString<'tcx> for ty::ExistentialBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        if self.builtin_bounds.contains(&ty::BoundSend) &&
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

impl<'tcx> UserString<'tcx> for ty::BuiltinBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        self.iter()
            .map(|bb| bb.user_string(tcx))
            .collect::<Vec<String>>()
            .connect("+")
            .to_string()
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitRef<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        // Replace any anonymous late-bound regions with named
        // variants, using gensym'd identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        let mut names = Vec::new();
        let (trait_ref, _) = ty::replace_late_bound_regions(tcx, self, |br, debruijn| {
            ty::ReLateBound(debruijn, match br {
                ty::BrNamed(_, name) => {
                    names.push(token::get_name(name));
                    br
                }
                ty::BrAnon(_) |
                ty::BrFresh(_) |
                ty::BrEnv => {
                    let name = token::gensym("r");
                    names.push(token::get_name(name));
                    ty::BrNamed(ast_util::local_def(ast::DUMMY_NODE_ID), name)
                }
            })
        });
        let names: Vec<_> = names.iter().map(|s| s.get()).collect();

        // Let the base string be either `SomeTrait` for `for<'a,'b> SomeTrait`,
        // depending on whether there are bound regions.
        let path_str = ty::item_path_str(tcx, self.def_id);
        let base =
            if names.is_empty() {
                path_str
            } else {
                format!("for<{}> {}", names.connect(","), path_str)
            };

        let trait_def = ty::lookup_trait_def(tcx, self.def_id);
        parameterized(tcx, base.as_slice(), &trait_ref.substs, &trait_def.generics)
    }
}

impl<'tcx> UserString<'tcx> for Ty<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        ty_to_string(tcx, *self)
    }
}

impl<'tcx> UserString<'tcx> for ast::Ident {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(self.name).get().to_string()
    }
}

impl<'tcx> Repr<'tcx> for abi::Abi {
    fn repr(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl<'tcx> UserString<'tcx> for abi::Abi {
    fn user_string(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarId {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarId({};`{}`;{})",
                self.var_id,
                ty::local_var_name_str(tcx, self.var_id),
                self.closure_expr_id)
    }
}

impl<'tcx> Repr<'tcx> for ast::Mutability {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BorrowKind {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarBorrow {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarBorrow({}, {})",
                self.kind.repr(tcx),
                self.region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::FloatVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::RegionVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TyVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVarValue {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::IntTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::UintTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::FloatTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::ExplicitSelfCategory {
    fn repr(&self, _: &ctxt) -> String {
        explicit_self_category_to_str(self).to_string()
    }
}


impl<'tcx> Repr<'tcx> for regionmanip::WfConstraint<'tcx> {
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

impl<'tcx> UserString<'tcx> for ParamTy {
    fn user_string(&self, tcx: &ctxt) -> String {
        let id = self.idx;
        let did = self.def_id;
        let ident = match tcx.ty_param_defs.borrow().get(&did.node) {
            Some(def) => token::get_name(def.name).get().to_string(),

            // This can only happen when a type mismatch error happens and
            // the actual type has more type parameters than the expected one.
            None => format!("<generic #{}>", id),
        };
        ident
    }
}

impl<'tcx> Repr<'tcx> for ParamTy {
    fn repr(&self, tcx: &ctxt) -> String {
        let ident = self.user_string(tcx);
        format!("{}/{}.{}", ident, self.space, self.idx)
    }
}

impl<'tcx, A:Repr<'tcx>, B:Repr<'tcx>> Repr<'tcx> for (A,B) {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let &(ref a, ref b) = self;
        format!("({},{})", a.repr(tcx), b.repr(tcx))
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for ty::Binder<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Binder({})", self.value.repr(tcx))
    }
}
