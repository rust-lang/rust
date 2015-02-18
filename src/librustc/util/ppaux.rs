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
use middle::region;
use middle::subst::{VecPerParamSpace,Subst};
use middle::subst;
use middle::ty::{BoundRegion, BrAnon, BrNamed};
use middle::ty::{ReEarlyBound, BrFresh, ctxt};
use middle::ty::{ReFree, ReScope, ReInfer, ReStatic, Region, ReEmpty};
use middle::ty::{ReSkolemized, ReVar, BrEnv};
use middle::ty::{mt, Ty, ParamTy};
use middle::ty::{ty_bool, ty_char, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_str, ty_vec, ty_float, ty_bare_fn};
use middle::ty::{ty_param, ty_ptr, ty_rptr, ty_tup, ty_open};
use middle::ty::{ty_closure};
use middle::ty::{ty_uniq, ty_trait, ty_int, ty_uint, ty_infer};
use middle::ty;
use middle::ty_fold::TypeFoldable;

use std::collections::HashMap;
use std::collections::hash_state::HashState;
use std::hash::Hash;
#[cfg(stage0)] use std::hash::Hasher;
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
pub trait Repr<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String;
}

/// Produces a string suitable for showing to the user.
pub trait UserString<'tcx> : Repr<'tcx> {
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
            &format!("{}{}{}", prefix, *str, suffix)[]);
        Some(span)
      }
      (ref str, None) => {
        cx.sess.note(
            &format!("{}{}{}", prefix, *str, suffix)[]);
        None
      }
    }
}

/// When a free region is associated with `item`, how should we describe the item in the error
/// message.
fn item_scope_tag(item: &ast::Item) -> &'static str {
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
        let new_string;
        let on_unknown_scope = || {
          (format!("unknown scope: {:?}.  Please report a bug.", scope), None)
        };
        let span = match scope.span(&cx.map) {
          Some(s) => s,
          None => return on_unknown_scope(),
        };
        let tag = match cx.map.find(scope.node_id()) {
          Some(ast_map::NodeBlock(_)) => "block",
          Some(ast_map::NodeExpr(expr)) => match expr.node {
              ast::ExprCall(..) => "call",
              ast::ExprMethodCall(..) => "method call",
              ast::ExprMatch(_, _, ast::MatchSource::IfLetDesugar { .. }) => "if let",
              ast::ExprMatch(_, _, ast::MatchSource::WhileLetDesugar) =>  "while let",
              ast::ExprMatch(_, _, ast::MatchSource::ForLoopDesugar) =>  "for",
              ast::ExprMatch(..) => "match",
              _ => "expression",
          },
          Some(ast_map::NodeStmt(_)) => "statement",
          Some(ast_map::NodeItem(it)) => item_scope_tag(&*it),
          Some(_) | None => {
            // this really should not happen
            return on_unknown_scope();
          }
        };
        let scope_decorated_tag = match scope {
            region::CodeExtent::Misc(_) => tag,
            region::CodeExtent::DestructionScope(_) => {
                new_string = format!("destruction scope surrounding {}", tag);
                new_string.as_slice()
            }
            region::CodeExtent::Remainder(r) => {
                new_string = format!("block suffix following statement {}",
                                     r.first_statement_index);
                &*new_string
            }
        };
        explain_span(cx, scope_decorated_tag, span)

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

        match cx.map.find(fr.scope.node_id) {
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
              (format!("{} unknown free region bounded by scope {:?}", prefix, fr.scope), None)
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
        (format!("the {} at {}:{}", heading, lo.line, lo.col.to_usize()),
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
            token::get_name(name).to_string()
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

pub fn vec_map_to_string<T, F>(ts: &[T], f: F) -> String where
    F: FnMut(&T) -> String,
{
    let tstrs = ts.iter().map(f).collect::<Vec<String>>();
    format!("[{}]", tstrs.connect(", "))
}

pub fn ty_to_string<'tcx>(cx: &ctxt<'tcx>, typ: &ty::TyS<'tcx>) -> String {
    fn bare_fn_to_string<'tcx>(cx: &ctxt<'tcx>,
                               opt_def_id: Option<ast::DefId>,
                               unsafety: ast::Unsafety,
                               abi: abi::Abi,
                               ident: Option<ast::Ident>,
                               sig: &ty::PolyFnSig<'tcx>)
                               -> String {
        let mut s = String::new();

        match unsafety {
            ast::Unsafety::Normal => {}
            ast::Unsafety::Unsafe => {
                s.push_str(&unsafety.to_string());
                s.push(' ');
            }
        };

        if abi != abi::Rust {
            s.push_str(&format!("extern {} ", abi.to_string())[]);
        };

        s.push_str("fn");

        match ident {
            Some(i) => {
                s.push(' ');
                s.push_str(&token::get_ident(i));
            }
            _ => { }
        }

        push_sig_to_string(cx, &mut s, '(', ')', sig);

        match opt_def_id {
            Some(def_id) => {
                s.push_str(" {");
                let path_str = ty::item_path_str(cx, def_id);
                s.push_str(&path_str[..]);
                s.push_str("}");
            }
            None => { }
        }

        s
    }

    fn closure_to_string<'tcx>(cx: &ctxt<'tcx>, cty: &ty::ClosureTy<'tcx>) -> String {
        let mut s = String::new();
        s.push_str("[closure");
        push_sig_to_string(cx, &mut s, '(', ')', &cty.sig);
        s.push(']');
        s
    }

    fn push_sig_to_string<'tcx>(cx: &ctxt<'tcx>,
                                s: &mut String,
                                bra: char,
                                ket: char,
                                sig: &ty::PolyFnSig<'tcx>) {
        s.push(bra);
        let strs = sig.0.inputs
            .iter()
            .map(|a| ty_to_string(cx, *a))
            .collect::<Vec<_>>();
        s.push_str(&strs.connect(", "));
        if sig.0.variadic {
            s.push_str(", ...");
        }
        s.push(ket);

        match sig.0.output {
            ty::FnConverging(t) => {
                if !ty::type_is_nil(t) {
                   s.push_str(" -> ");
                   s.push_str(&ty_to_string(cx, t)[]);
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
            ty::FreshTy(v) => format!("FreshTy({})", v),
            ty::FreshIntTy(v) => format!("FreshIntTy({})", v)
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
            let mut buf = region_ptr_to_string(cx, *r);
            buf.push_str(&mt_to_string(cx, tm)[]);
            buf
        }
        ty_open(typ) =>
            format!("opened<{}>", ty_to_string(cx, typ)),
        ty_tup(ref elems) => {
            let strs = elems
                .iter()
                .map(|elem| ty_to_string(cx, *elem))
                .collect::<Vec<_>>();
            match &strs[..] {
                [ref string] => format!("({},)", string),
                strs => format!("({})", strs.connect(", "))
            }
        }
        ty_bare_fn(opt_def_id, ref f) => {
            bare_fn_to_string(cx, opt_def_id, f.unsafety, f.abi, None, &f.sig)
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
        ty_enum(did, substs) | ty_struct(did, substs) => {
            let base = ty::item_path_str(cx, did);
            parameterized(cx, &base, substs, did, &[],
                          || ty::lookup_item_type(cx, did).generics)
        }
        ty_trait(ref data) => {
            data.user_string(cx)
        }
        ty::ty_projection(ref data) => {
            format!("<{} as {}>::{}",
                    data.trait_ref.self_ty().user_string(cx),
                    data.trait_ref.user_string(cx),
                    data.item_name.user_string(cx))
        }
        ty_str => "str".to_string(),
        ty_closure(ref did, _, substs) => {
            let closure_tys = cx.closure_tys.borrow();
            closure_tys.get(did).map(|closure_type| {
                closure_to_string(cx, &closure_type.subst(cx, substs))
            }).unwrap_or_else(|| {
                if did.krate == ast::LOCAL_CRATE {
                    let span = cx.map.span(did.node);
                    format!("[closure {}]", span.repr(cx))
                } else {
                    format!("[closure]")
                }
            })
        }
        ty_vec(t, sz) => {
            let inner_str = ty_to_string(cx, t);
            match sz {
                Some(n) => format!("[{}; {}]", inner_str, n),
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

pub fn parameterized<'tcx,GG>(cx: &ctxt<'tcx>,
                              base: &str,
                              substs: &subst::Substs<'tcx>,
                              did: ast::DefId,
                              projections: &[ty::ProjectionPredicate<'tcx>],
                              get_generics: GG)
                              -> String
    where GG : FnOnce() -> ty::Generics<'tcx>
{
    if cx.sess.verbose() {
        let mut strings = vec![];
        match substs.regions {
            subst::ErasedRegions => {
                strings.push(format!(".."));
            }
            subst::NonerasedRegions(ref regions) => {
                for region in regions.iter() {
                    strings.push(region.repr(cx));
                }
            }
        }
        for ty in substs.types.iter() {
            strings.push(ty.repr(cx));
        }
        for projection in projections.iter() {
            strings.push(format!("{}={}",
                                 projection.projection_ty.item_name.user_string(cx),
                                 projection.ty.user_string(cx)));
        }
        return if strings.is_empty() {
            format!("{}", base)
        } else {
            format!("{}<{}>", base, strings.connect(","))
        };
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

    // It is important to execute this conditionally, only if -Z
    // verbose is false. Otherwise, debug logs can sometimes cause
    // ICEs trying to fetch the generics early in the pipeline. This
    // is kind of a hacky workaround in that -Z verbose is required to
    // avoid those ICEs.
    let generics = get_generics();

    let has_self = substs.self_ty().is_some();
    let tps = substs.types.get_slice(subst::TypeSpace);
    let ty_params = generics.types.get_slice(subst::TypeSpace);
    let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
    let num_defaults = if has_defaults {
        ty_params.iter().zip(tps.iter()).rev().take_while(|&(def, &actual)| {
            match def.default {
                Some(default) => {
                    if !has_self && ty::type_has_self(default) {
                        // In an object type, there is no `Self`, and
                        // thus if the default value references Self,
                        // the user will be required to give an
                        // explicit value. We can't even do the
                        // substitution below to check without causing
                        // an ICE. (#18956).
                        false
                    } else {
                        default.subst(cx, substs) == actual
                    }
                }
                None => false
            }
        }).count()
    } else {
        0
    };

    for t in &tps[..tps.len() - num_defaults] {
        strs.push(ty_to_string(cx, *t))
    }

    for projection in projections {
        strs.push(format!("{}={}",
                          projection.projection_ty.item_name.user_string(cx),
                          projection.ty.user_string(cx)));
    }

    if cx.lang_items.fn_trait_kind(did).is_some() && projections.len() == 1 {
        let projection_ty = projections[0].ty;
        let tail =
            if ty::type_is_nil(projection_ty) {
                format!("")
            } else {
                format!(" -> {}", projection_ty.user_string(cx))
            };
        format!("{}({}){}",
                base,
                if strs[0].starts_with("(") && strs[0].ends_with(",)") {
                    &strs[0][1 .. strs[0].len() - 2] // Remove '(' and ',)'
                } else if strs[0].starts_with("(") && strs[0].ends_with(")") {
                    &strs[0][1 .. strs[0].len() - 1] // Remove '(' and ')'
                } else {
                    &strs[0][]
                },
                tail)
    } else if strs.len() > 0 {
        format!("{}<{}>", base, strs.connect(", "))
    } else {
        format!("{}", base)
    }
}

pub fn ty_to_short_str<'tcx>(cx: &ctxt<'tcx>, typ: Ty<'tcx>) -> String {
    let mut s = typ.repr(cx).to_string();
    if s.len() >= 32 {
        s = (&s[0..32]).to_string();
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
        (**self).repr(tcx)
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

impl<'a, 'tcx, T: ?Sized +Repr<'tcx>> Repr<'tcx> for &'a T {
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
        repr_vec(tcx, &self[..])
    }
}

// This is necessary to handle types like Option<~[T]>, for which
// autoderef cannot convert the &[T] handler
impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Vec<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        repr_vec(tcx, &self[..])
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
        format!("{:?}", *self)
    }
}

/// This curious type is here to help pretty-print trait objects. In
/// a trait object, the projections are stored separately from the
/// main trait bound, but in fact we want to package them together
/// when printing out; they also have separate binders, but we want
/// them to share a binder when we print them out. (And the binder
/// pretty-printing logic is kind of clever and we don't want to
/// reproduce it.) So we just repackage up the structure somewhat.
///
/// Right now there is only one trait in an object that can have
/// projection bounds, so we just stuff them altogether. But in
/// reality we should eventually sort things out better.
type TraitAndProjections<'tcx> =
    (Rc<ty::TraitRef<'tcx>>, Vec<ty::ProjectionPredicate<'tcx>>);

impl<'tcx> UserString<'tcx> for TraitAndProjections<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let &(ref trait_ref, ref projection_bounds) = self;
        let base = ty::item_path_str(tcx, trait_ref.def_id);
        parameterized(tcx,
                      &base,
                      trait_ref.substs,
                      trait_ref.def_id,
                      &projection_bounds[..],
                      || ty::lookup_trait_def(tcx, trait_ref.def_id).generics.clone())
    }
}

impl<'tcx> UserString<'tcx> for ty::TyTrait<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let &ty::TyTrait { ref principal, ref bounds } = self;

        let mut components = vec![];

        let tap: ty::Binder<TraitAndProjections<'tcx>> =
            ty::Binder((principal.0.clone(),
                        bounds.projection_bounds.iter().map(|x| x.0.clone()).collect()));

        // Generate the main trait ref, including associated types.
        components.push(tap.user_string(tcx));

        // Builtin bounds.
        for bound in &bounds.builtin_bounds {
            components.push(bound.user_string(tcx));
        }

        // Region, if not obviously implied by builtin bounds.
        if bounds.region_bound != ty::ReStatic {
            // Region bound is implied by builtin bounds:
            components.push(bounds.region_bound.user_string(tcx));
        }

        components.retain(|s| !s.is_empty());

        components.connect(" + ")
    }
}

impl<'tcx> Repr<'tcx> for ty::TypeParameterDef<'tcx> {
    fn repr(&self, _tcx: &ctxt<'tcx>) -> String {
        format!("TypeParameterDef({:?}, {:?}/{})",
                self.def_id,
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
        format!("[{};{};{}]",
                self.get_slice(subst::TypeSpace).repr(tcx),
                self.get_slice(subst::SelfSpace).repr(tcx),
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
        for b in self {
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

impl<'tcx> Repr<'tcx> for ty::ParamBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let mut res = Vec::new();
        res.push(self.builtin_bounds.repr(tcx));
        for t in &self.trait_bounds {
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
        parameterized(tcx, &base, self.substs, self.def_id, &[],
                      || ty::lookup_trait_def(tcx, self.def_id).generics.clone())
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
                format!("ReEarlyBound({}, {:?}, {}, {})",
                               id,
                               space,
                               index,
                               token::get_name(name))
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                format!("ReLateBound({:?}, {})",
                        binder_id,
                        bound_region.repr(tcx))
            }

            ty::ReFree(ref fr) => fr.repr(tcx),

            ty::ReScope(id) => {
                format!("ReScope({:?})", id)
            }

            ty::ReStatic => {
                "ReStatic".to_string()
            }

            ty::ReInfer(ReVar(ref vid)) => {
                format!("{:?}", vid)
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
                self.scope.repr(tcx),
                self.bound_region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for region::CodeExtent {
    fn repr(&self, _tcx: &ctxt) -> String {
        match *self {
            region::CodeExtent::Misc(node_id) =>
                format!("Misc({})", node_id),
            region::CodeExtent::DestructionScope(node_id) =>
                format!("DestructionScope({})", node_id),
            region::CodeExtent::Remainder(rem) =>
                format!("Remainder({}, {})", rem.block, rem.first_statement_index),
        }
    }
}

impl<'tcx> Repr<'tcx> for region::DestructionScopeData {
    fn repr(&self, _tcx: &ctxt) -> String {
        match *self {
            region::DestructionScopeData{ node_id } =>
                format!("DestructionScopeData {{ node_id: {} }}", node_id),
        }
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
                                "{:?}:{}",
                                *self,
                                ty::item_path_str(tcx, *self))
                }
                _ => {}
            }
        }
        return format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TypeScheme<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TypeScheme {{generics: {}, ty: {}}}",
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

impl<'tcx> Repr<'tcx> for ty::GenericPredicates<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("GenericPredicates(predicates: {})",
                self.predicates.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::InstantiatedPredicates<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("InstantiatedPredicates({})",
                self.predicates.repr(tcx))
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
        token::get_name(*self).to_string()
    }
}

impl<'tcx> UserString<'tcx> for ast::Name {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::Ident {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_ident(*self).to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::ExplicitSelf_ {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::Visibility {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BareFnTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("BareFnTy {{unsafety: {}, abi: {}, sig: {}}}",
                self.unsafety,
                self.abi.to_string(),
                self.sig.repr(tcx))
    }
}


impl<'tcx> Repr<'tcx> for ty::FnSig<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("fn{} -> {}", self.inputs.repr(tcx), self.output.repr(tcx))
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

impl<'tcx> Repr<'tcx> for ty::MethodCallee<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodCallee {{origin: {}, ty: {}, {}}}",
                self.origin.repr(tcx),
                self.ty.repr(tcx),
                self.substs.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodOrigin<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &ty::MethodStatic(def_id) => {
                format!("MethodStatic({})", def_id.repr(tcx))
            }
            &ty::MethodStaticClosure(def_id) => {
                format!("MethodStaticClosure({})", def_id.repr(tcx))
            }
            &ty::MethodTypeParam(ref p) => {
                p.repr(tcx)
            }
            &ty::MethodTraitObject(ref p) => {
                p.repr(tcx)
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodParam<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodParam({},{})",
                self.trait_ref.repr(tcx),
                self.method_num)
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodObject<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodObject({},{},{})",
                self.trait_ref.repr(tcx),
                self.method_num,
                self.vtable_index)
    }
}

impl<'tcx> Repr<'tcx> for ty::BuiltinBound {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
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
        for n in &self.trait_bounds {
            result.push(n.user_string(tcx));
        }
        result.connect(" + ")
    }
}

impl<'tcx> Repr<'tcx> for ty::ExistentialBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let mut res = Vec::new();

        let region_str = self.region_bound.user_string(tcx);
        if !region_str.is_empty() {
            res.push(region_str);
        }

        for bound in &self.builtin_bounds {
            res.push(bound.user_string(tcx));
        }

        for projection_bound in &self.projection_bounds {
            res.push(projection_bound.user_string(tcx));
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

impl<'tcx, T> UserString<'tcx> for ty::Binder<T>
    where T : UserString<'tcx> + TypeFoldable<'tcx>
{
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        // Replace any anonymous late-bound regions with named
        // variants, using gensym'd identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        let mut names = Vec::new();
        let (unbound_value, _) = ty::replace_late_bound_regions(tcx, self, |br| {
            ty::ReLateBound(ty::DebruijnIndex::new(1), match br {
                ty::BrNamed(_, name) => {
                    names.push(token::get_name(name));
                    br
                }
                ty::BrAnon(_) |
                ty::BrFresh(_) |
                ty::BrEnv => {
                    let name = token::gensym("'r");
                    names.push(token::get_name(name));
                    ty::BrNamed(ast_util::local_def(ast::DUMMY_NODE_ID), name)
                }
            })
        });
        let names: Vec<_> = names.iter().map(|s| &s[..]).collect();

        let value_str = unbound_value.user_string(tcx);
        if names.len() == 0 {
            value_str
        } else {
            format!("for<{}> {}", names.connect(","), value_str)
        }
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitRef<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let path_str = ty::item_path_str(tcx, self.def_id);
        parameterized(tcx, &path_str, self.substs, self.def_id, &[],
                      || ty::lookup_trait_def(tcx, self.def_id).generics.clone())
    }
}

impl<'tcx> UserString<'tcx> for Ty<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        ty_to_string(tcx, *self)
    }
}

impl<'tcx> UserString<'tcx> for ast::Ident {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(self.name).to_string()
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
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BorrowKind {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarBorrow {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarBorrow({}, {})",
                self.kind.repr(tcx),
                self.region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarCapture {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::UpvarCapture::ByValue => format!("ByValue"),
            ty::UpvarCapture::ByRef(ref data) => format!("ByRef({})", data.repr(tcx)),
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::FloatVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::RegionVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TyVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVarValue {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::IntTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::UintTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::FloatTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::ExplicitSelfCategory {
    fn repr(&self, _: &ctxt) -> String {
        explicit_self_category_to_str(self).to_string()
    }
}

impl<'tcx> UserString<'tcx> for ParamTy {
    fn user_string(&self, _tcx: &ctxt) -> String {
        format!("{}", token::get_name(self.name))
    }
}

impl<'tcx> Repr<'tcx> for ParamTy {
    fn repr(&self, tcx: &ctxt) -> String {
        let ident = self.user_string(tcx);
        format!("{}/{:?}.{}", ident, self.space, self.idx)
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
        format!("Binder({})", self.0.repr(tcx))
    }
}

#[cfg(stage0)]
impl<'tcx, S, K, V> Repr<'tcx> for HashMap<K, V, S>
    where K: Hash<<S as HashState>::Hasher> + Eq + Repr<'tcx>,
          V: Repr<'tcx>,
          S: HashState,
          <S as HashState>::Hasher: Hasher<Output=u64>,
{
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("HashMap({})",
                self.iter()
                    .map(|(k,v)| format!("{} => {}", k.repr(tcx), v.repr(tcx)))
                    .collect::<Vec<String>>()
                    .connect(", "))
    }
}

#[cfg(not(stage0))]
impl<'tcx, S, K, V> Repr<'tcx> for HashMap<K, V, S>
    where K: Hash + Eq + Repr<'tcx>,
          V: Repr<'tcx>,
          S: HashState,
{
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("HashMap({})",
                self.iter()
                    .map(|(k,v)| format!("{} => {}", k.repr(tcx), v.repr(tcx)))
                    .collect::<Vec<String>>()
                    .connect(", "))
    }
}

impl<'tcx, T, U> Repr<'tcx> for ty::OutlivesPredicate<T,U>
    where T : Repr<'tcx> + TypeFoldable<'tcx>,
          U : Repr<'tcx> + TypeFoldable<'tcx>,
{
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("OutlivesPredicate({}, {})",
                self.0.repr(tcx),
                self.1.repr(tcx))
    }
}

impl<'tcx, T, U> UserString<'tcx> for ty::OutlivesPredicate<T,U>
    where T : UserString<'tcx> + TypeFoldable<'tcx>,
          U : UserString<'tcx> + TypeFoldable<'tcx>,
{
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} : {}",
                self.0.user_string(tcx),
                self.1.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::EquatePredicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("EquatePredicate({}, {})",
                self.0.repr(tcx),
                self.1.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::EquatePredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} == {}",
                self.0.user_string(tcx),
                self.1.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitPredicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TraitPredicate({})",
                self.trait_ref.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitPredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} : {}",
                self.trait_ref.self_ty().user_string(tcx),
                self.trait_ref.user_string(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} == {}",
                self.projection_ty.user_string(tcx),
                self.ty.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ProjectionTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("<{} as {}>::{}",
                self.trait_ref.substs.self_ty().repr(tcx),
                self.trait_ref.repr(tcx),
                self.item_name.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::ProjectionTy<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("<{} as {}>::{}",
                self.trait_ref.self_ty().user_string(tcx),
                self.trait_ref.user_string(tcx),
                self.item_name.user_string(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::Predicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            ty::Predicate::Trait(ref data) => data.user_string(tcx),
            ty::Predicate::Equate(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::RegionOutlives(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::TypeOutlives(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::Projection(ref predicate) => predicate.user_string(tcx),
        }
    }
}
