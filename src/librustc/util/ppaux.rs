// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use ast_map;
use middle::def;
use middle::region;
use middle::subst::{VecPerParamSpace,Subst};
use middle::subst;
use middle::ty::{BoundRegion, BrAnon, BrNamed};
use middle::ty::{ReEarlyBound, BrFresh, ctxt};
use middle::ty::{ReFree, ReScope, ReInfer, ReStatic, Region, ReEmpty};
use middle::ty::{ReSkolemized, ReVar, BrEnv};
use middle::ty::{mt, Ty, ParamTy};
use middle::ty::{TyBool, TyChar, TyStruct, TyEnum};
use middle::ty::{TyError, TyStr, TyArray, TySlice, TyFloat, TyBareFn};
use middle::ty::{TyParam, TyRawPtr, TyRef, TyTuple};
use middle::ty::TyClosure;
use middle::ty::{TyBox, TyTrait, TyInt, TyUint, TyInfer};
use middle::ty;
use middle::ty_fold::{self, TypeFoldable};

use std::collections::HashMap;
use std::collections::hash_state::HashState;
use std::hash::Hash;
use std::rc::Rc;
use syntax::abi;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::{ast, ast_util};
use syntax::owned_slice::OwnedSlice;

/// Produces a string suitable for debugging output.
pub trait Repr {
    fn repr(&self) -> String;
}

/// Produces a string suitable for showing to the user.
pub trait UserString: Repr {
    fn user_string(&self) -> String;
}

pub fn verbose() -> bool {
    ty::tls::with(|tcx| tcx.sess.verbose())
}

fn parameterized<GG>(substs: &subst::Substs,
                     did: ast::DefId,
                     projections: &[ty::ProjectionPredicate],
                     get_generics: GG)
                     -> String
    where GG: for<'tcx> FnOnce(&ty::ctxt<'tcx>) -> ty::Generics<'tcx>
{
    let base = ty::tls::with(|tcx| ty::item_path_str(tcx, did));
    if verbose() {
        let mut strings = vec![];
        match substs.regions {
            subst::ErasedRegions => {
                strings.push(format!(".."));
            }
            subst::NonerasedRegions(ref regions) => {
                for region in regions {
                    strings.push(region.repr());
                }
            }
        }
        for ty in &substs.types {
            strings.push(ty.repr());
        }
        for projection in projections {
            strings.push(format!("{}={}",
                                 projection.projection_ty.item_name.user_string(),
                                 projection.ty.user_string()));
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
            for &r in regions {
                let s = r.user_string();
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
    ty::tls::with(|tcx| {
        let generics = get_generics(tcx);

        let has_self = substs.self_ty().is_some();
        let tps = substs.types.get_slice(subst::TypeSpace);
        let ty_params = generics.types.get_slice(subst::TypeSpace);
        let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
        let num_defaults = if has_defaults {
            let substs = tcx.lift(&substs);
            ty_params.iter().zip(tps).rev().take_while(|&(def, &actual)| {
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
                            let default = tcx.lift(&default);
                            substs.and_then(|substs| default.subst(tcx, substs)) == Some(actual)
                        }
                    }
                    None => false
                }
            }).count()
        } else {
            0
        };

        for t in &tps[..tps.len() - num_defaults] {
            strs.push(t.user_string())
        }
    });

    for projection in projections {
        strs.push(format!("{}={}",
                          projection.projection_ty.item_name.user_string(),
                          projection.ty.user_string()));
    }

    let fn_trait_kind = ty::tls::with(|tcx| tcx.lang_items.fn_trait_kind(did));
    if fn_trait_kind.is_some() && projections.len() == 1 {
        let projection_ty = projections[0].ty;
        let tail =
            if ty::type_is_nil(projection_ty) {
                format!("")
            } else {
                format!(" -> {}", projection_ty.user_string())
            };
        format!("{}({}){}",
                base,
                if strs[0].starts_with("(") && strs[0].ends_with(",)") {
                    &strs[0][1 .. strs[0].len() - 2] // Remove '(' and ',)'
                } else if strs[0].starts_with("(") && strs[0].ends_with(")") {
                    &strs[0][1 .. strs[0].len() - 1] // Remove '(' and ')'
                } else {
                    &strs[0][..]
                },
                tail)
    } else if !strs.is_empty() {
        format!("{}<{}>", base, strs.connect(", "))
    } else {
        format!("{}", base)
    }
}

fn in_binder<'tcx, T, U>(tcx: &ty::ctxt<'tcx>,
                         original: &ty::Binder<T>,
                         lifted: Option<ty::Binder<U>>) -> String
    where T: UserString, U: UserString + TypeFoldable<'tcx>
{
    // Replace any anonymous late-bound regions with named
    // variants, using gensym'd identifiers, so that we can
    // clearly differentiate between named and unnamed regions in
    // the output. We'll probably want to tweak this over time to
    // decide just how much information to give.
    let value = if let Some(v) = lifted {
        v
    } else {
        return original.0.user_string();
    };
    let mut names = Vec::new();
    let value_str = ty_fold::replace_late_bound_regions(tcx, &value, |br| {
        ty::ReLateBound(ty::DebruijnIndex::new(1), match br {
            ty::BrNamed(_, name) => {
                names.push(token::get_name(name).to_string());
                br
            }
            ty::BrAnon(_) |
            ty::BrFresh(_) |
            ty::BrEnv => {
                let name = token::gensym("'r");
                names.push(token::get_name(name).to_string());
                ty::BrNamed(ast_util::local_def(ast::DUMMY_NODE_ID), name)
            }
        })
    }).0.user_string();

    if names.is_empty() {
        value_str
    } else {
        format!("for<{}> {}", names.connect(","), value_str)
    }
}

impl<T:Repr> Repr for Option<T> {
    fn repr(&self) -> String {
        match self {
            &None => "None".to_string(),
            &Some(ref t) => t.repr(),
        }
    }
}

impl<T:Repr> Repr for P<T> {
    fn repr(&self) -> String {
        (**self).repr()
    }
}

impl<T:Repr,U:Repr> Repr for Result<T,U> {
    fn repr(&self) -> String {
        match self {
            &Ok(ref t) => t.repr(),
            &Err(ref u) => format!("Err({})", u.repr())
        }
    }
}

impl Repr for () {
    fn repr(&self) -> String {
        "()".to_string()
    }
}

impl<'a, T: ?Sized +Repr> Repr for &'a T {
    fn repr(&self) -> String {
        (**self).repr()
    }
}

impl<T:Repr> Repr for Rc<T> {
    fn repr(&self) -> String {
        (&**self).repr()
    }
}

impl<T:Repr> Repr for Box<T> {
    fn repr(&self) -> String {
        (&**self).repr()
    }
}

impl<T:Repr> Repr for [T] {
    fn repr(&self) -> String {
        format!("[{}]", self.iter().map(|t| t.repr()).collect::<Vec<_>>().connect(", "))
    }
}

impl<T:Repr> Repr for OwnedSlice<T> {
    fn repr(&self) -> String {
        self[..].repr()
    }
}

// This is necessary to handle types like Option<Vec<T>>, for which
// autoderef cannot convert the &[T] handler
impl<T:Repr> Repr for Vec<T> {
    fn repr(&self) -> String {
        self[..].repr()
    }
}

impl<'a, T: ?Sized +UserString> UserString for &'a T {
    fn user_string(&self) -> String {
        (**self).user_string()
    }
}

impl<T:UserString> UserString for Vec<T> {
    fn user_string(&self) -> String {
        let strs: Vec<String> =
            self.iter().map(|t| t.user_string()).collect();
        strs.connect(", ")
    }
}

impl Repr for def::Def {
    fn repr(&self) -> String {
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
    (ty::TraitRef<'tcx>, Vec<ty::ProjectionPredicate<'tcx>>);

impl<'tcx> UserString for TraitAndProjections<'tcx> {
    fn user_string(&self) -> String {
        let &(ref trait_ref, ref projection_bounds) = self;
        parameterized(trait_ref.substs,
                      trait_ref.def_id,
                      &projection_bounds[..],
                      |tcx| ty::lookup_trait_def(tcx, trait_ref.def_id).generics.clone())
    }
}

impl<'tcx> UserString for ty::TraitTy<'tcx> {
    fn user_string(&self) -> String {
        let &ty::TraitTy { ref principal, ref bounds } = self;

        let mut components = vec![];

        let tap: ty::Binder<TraitAndProjections<'tcx>> =
            ty::Binder((principal.0.clone(),
                        bounds.projection_bounds.iter().map(|x| x.0.clone()).collect()));

        // Generate the main trait ref, including associated types.
        components.push(tap.user_string());

        // Builtin bounds.
        for bound in &bounds.builtin_bounds {
            components.push(bound.user_string());
        }

        // Region, if not obviously implied by builtin bounds.
        if bounds.region_bound != ty::ReStatic {
            // Region bound is implied by builtin bounds:
            components.push(bounds.region_bound.user_string());
        }

        components.retain(|s| !s.is_empty());

        components.connect(" + ")
    }
}

impl<'tcx> Repr for ty::TypeParameterDef<'tcx> {
    fn repr(&self) -> String {
        format!("TypeParameterDef({:?}, {:?}/{})",
                self.def_id,
                self.space,
                self.index)
    }
}

impl Repr for ty::RegionParameterDef {
    fn repr(&self) -> String {
        format!("RegionParameterDef(name={}, def_id={}, bounds={})",
                token::get_name(self.name),
                self.def_id.repr(),
                self.bounds.repr())
    }
}

impl<'tcx> Repr for ty::TyS<'tcx> {
    fn repr(&self) -> String {
        self.user_string()
    }
}

impl<'tcx> Repr for ty::mt<'tcx> {
    fn repr(&self) -> String {
        format!("{}{}",
            if self.mutbl == ast::MutMutable { "mut " } else { "" },
            self.ty.user_string())
    }
}

impl<'tcx> Repr for subst::Substs<'tcx> {
    fn repr(&self) -> String {
        format!("Substs[types={}, regions={}]",
                       self.types.repr(),
                       self.regions.repr())
    }
}

impl<T:Repr> Repr for subst::VecPerParamSpace<T> {
    fn repr(&self) -> String {
        format!("[{};{};{}]",
                self.get_slice(subst::TypeSpace).repr(),
                self.get_slice(subst::SelfSpace).repr(),
                self.get_slice(subst::FnSpace).repr())
    }
}

impl<'tcx> Repr for ty::ItemSubsts<'tcx> {
    fn repr(&self) -> String {
        format!("ItemSubsts({})", self.substs.repr())
    }
}

impl Repr for subst::RegionSubsts {
    fn repr(&self) -> String {
        match *self {
            subst::ErasedRegions => "erased".to_string(),
            subst::NonerasedRegions(ref regions) => regions.repr()
        }
    }
}

impl Repr for ty::BuiltinBounds {
    fn repr(&self) -> String {
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

impl<'tcx> Repr for ty::ParamBounds<'tcx> {
    fn repr(&self) -> String {
        let mut res = Vec::new();
        res.push(self.builtin_bounds.repr());
        for t in &self.trait_bounds {
            res.push(t.repr());
        }
        res.connect("+")
    }
}

impl<'tcx> Repr for ty::TraitRef<'tcx> {
    fn repr(&self) -> String {
        // when printing out the debug representation, we don't need
        // to enumerate the `for<...>` etc because the debruijn index
        // tells you everything you need to know.
        let result = self.user_string();
        match self.substs.self_ty() {
            None => result,
            Some(sty) => format!("<{} as {}>", sty.repr(), result)
        }
    }
}

impl<'tcx> Repr for ty::TraitDef<'tcx> {
    fn repr(&self) -> String {
        format!("TraitDef(generics={}, trait_ref={})",
                self.generics.repr(),
                self.trait_ref.repr())
    }
}

impl Repr for ast::TraitItem {
    fn repr(&self) -> String {
        let kind = match self.node {
            ast::ConstTraitItem(..) => "ConstTraitItem",
            ast::MethodTraitItem(..) => "MethodTraitItem",
            ast::TypeTraitItem(..) => "TypeTraitItem",
        };
        format!("{}({}, id={})", kind, self.ident, self.id)
    }
}

impl Repr for ast::Expr {
    fn repr(&self) -> String {
        format!("expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

impl Repr for ast::Path {
    fn repr(&self) -> String {
        format!("path({})", pprust::path_to_string(self))
    }
}

impl UserString for ast::Path {
    fn user_string(&self) -> String {
        pprust::path_to_string(self)
    }
}

impl Repr for ast::Ty {
    fn repr(&self) -> String {
        format!("type({})", pprust::ty_to_string(self))
    }
}

impl Repr for ast::Item {
    fn repr(&self) -> String {
        format!("item({})", ty::tls::with(|tcx| tcx.map.node_to_string(self.id)))
    }
}

impl Repr for ast::Lifetime {
    fn repr(&self) -> String {
        format!("lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

impl Repr for ast::Stmt {
    fn repr(&self) -> String {
        format!("stmt({}: {})",
                ast_util::stmt_id(self),
                pprust::stmt_to_string(self))
    }
}

impl Repr for ast::Pat {
    fn repr(&self) -> String {
        format!("pat({}: {})", self.id, pprust::pat_to_string(self))
    }
}

impl Repr for ty::BoundRegion {
    fn repr(&self) -> String {
        match *self {
            ty::BrAnon(id) => format!("BrAnon({})", id),
            ty::BrNamed(id, name) => {
                format!("BrNamed({}, {})", id.repr(), token::get_name(name))
            }
            ty::BrFresh(id) => format!("BrFresh({})", id),
            ty::BrEnv => "BrEnv".to_string()
        }
    }
}

impl UserString for ty::BoundRegion {
    fn user_string(&self) -> String {
        if verbose() {
            return self.repr();
        }

        match *self {
            BrNamed(_, name) => token::get_name(name).to_string(),
            BrAnon(_) | BrFresh(_) | BrEnv => String::new()
        }
    }
}

impl Repr for ty::Region {
    fn repr(&self) -> String {
        match *self {
            ty::ReEarlyBound(ref data) => {
                format!("ReEarlyBound({}, {:?}, {}, {})",
                        data.param_id,
                        data.space,
                        data.index,
                        token::get_name(data.name))
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                format!("ReLateBound({:?}, {})",
                        binder_id,
                        bound_region.repr())
            }

            ty::ReFree(ref fr) => fr.repr(),

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
                format!("re_skolemized({}, {})", id, bound_region.repr())
            }

            ty::ReEmpty => {
                "ReEmpty".to_string()
            }
        }
    }
}

impl UserString for ty::Region {
    fn user_string(&self) -> String {
        if verbose() {
            return self.repr();
        }

        // These printouts are concise.  They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string.  Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *self {
            ty::ReEarlyBound(ref data) => {
                token::get_name(data.name).to_string()
            }
            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) |
            ty::ReInfer(ReSkolemized(_, br)) => {
                br.user_string()
            }
            ty::ReScope(_) |
            ty::ReInfer(ReVar(_)) => String::new(),
            ty::ReStatic => "'static".to_owned(),
            ty::ReEmpty => "'<empty>".to_owned(),
        }
    }
}

impl Repr for ty::FreeRegion {
    fn repr(&self) -> String {
        format!("ReFree({}, {})",
                self.scope.repr(),
                self.bound_region.repr())
    }
}

impl Repr for region::CodeExtent {
    fn repr(&self) -> String {
        match *self {
            region::CodeExtent::ParameterScope { fn_id, body_id } =>
                format!("ParameterScope({}, {})", fn_id, body_id),
            region::CodeExtent::Misc(node_id) =>
                format!("Misc({})", node_id),
            region::CodeExtent::DestructionScope(node_id) =>
                format!("DestructionScope({})", node_id),
            region::CodeExtent::Remainder(rem) =>
                format!("Remainder({}, {})", rem.block, rem.first_statement_index),
        }
    }
}

impl Repr for region::DestructionScopeData {
    fn repr(&self) -> String {
        match *self {
            region::DestructionScopeData{ node_id } =>
                format!("DestructionScopeData {{ node_id: {} }}", node_id),
        }
    }
}

impl Repr for ast::DefId {
    fn repr(&self) -> String {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        ty::tls::with(|tcx| {
            if self.krate == ast::LOCAL_CRATE {
                match tcx.map.find(self.node) {
                    Some(ast_map::NodeItem(..)) |
                    Some(ast_map::NodeForeignItem(..)) |
                    Some(ast_map::NodeImplItem(..)) |
                    Some(ast_map::NodeTraitItem(..)) |
                    Some(ast_map::NodeVariant(..)) |
                    Some(ast_map::NodeStructCtor(..)) => {
                        return format!("{:?}:{}",
                                       *self,
                                       ty::item_path_str(tcx, *self));
                    }
                    _ => {}
                }
            }
            format!("{:?}", *self)
        })
    }
}

impl<'tcx> Repr for ty::TypeScheme<'tcx> {
    fn repr(&self) -> String {
        format!("TypeScheme {{generics: {}, ty: {}}}",
                self.generics.repr(),
                self.ty.repr())
    }
}

impl<'tcx> Repr for ty::Generics<'tcx> {
    fn repr(&self) -> String {
        format!("Generics(types: {}, regions: {})",
                self.types.repr(),
                self.regions.repr())
    }
}

impl<'tcx> Repr for ty::GenericPredicates<'tcx> {
    fn repr(&self) -> String {
        format!("GenericPredicates(predicates: {})",
                self.predicates.repr())
    }
}

impl<'tcx> Repr for ty::InstantiatedPredicates<'tcx> {
    fn repr(&self) -> String {
        format!("InstantiatedPredicates({})",
                self.predicates.repr())
    }
}

impl Repr for ty::ItemVariances {
    fn repr(&self) -> String {
        format!("ItemVariances(types={}, \
                regions={})",
                self.types.repr(),
                self.regions.repr())
    }
}

impl Repr for ty::Variance {
    fn repr(&self) -> String {
        // The first `.to_string()` returns a &'static str (it is not an implementation
        // of the ToString trait). Because of that, we need to call `.to_string()` again
        // if we want to have a `String`.
        let result: &'static str = (*self).to_string();
        result.to_string()
    }
}

impl<'tcx> Repr for ty::ImplOrTraitItem<'tcx> {
    fn repr(&self) -> String {
        format!("ImplOrTraitItem({})",
                match *self {
                    ty::ImplOrTraitItem::MethodTraitItem(ref i) => i.repr(),
                    ty::ImplOrTraitItem::ConstTraitItem(ref i) => i.repr(),
                    ty::ImplOrTraitItem::TypeTraitItem(ref i) => i.repr(),
                })
    }
}

impl<'tcx> Repr for ty::AssociatedConst<'tcx> {
    fn repr(&self) -> String {
        format!("AssociatedConst(name: {}, ty: {}, vis: {}, def_id: {})",
                self.name.repr(),
                self.ty.repr(),
                self.vis.repr(),
                self.def_id.repr())
    }
}

impl<'tcx> Repr for ty::AssociatedType<'tcx> {
    fn repr(&self) -> String {
        format!("AssociatedType(name: {}, vis: {}, def_id: {})",
                self.name.repr(),
                self.vis.repr(),
                self.def_id.repr())
    }
}

impl<'tcx> Repr for ty::Method<'tcx> {
    fn repr(&self) -> String {
        format!("Method(name: {}, generics: {}, predicates: {}, fty: {}, \
                 explicit_self: {}, vis: {}, def_id: {})",
                self.name.repr(),
                self.generics.repr(),
                self.predicates.repr(),
                self.fty.repr(),
                self.explicit_self.repr(),
                self.vis.repr(),
                self.def_id.repr())
    }
}

impl Repr for ast::Name {
    fn repr(&self) -> String {
        token::get_name(*self).to_string()
    }
}

impl UserString for ast::Name {
    fn user_string(&self) -> String {
        token::get_name(*self).to_string()
    }
}

impl Repr for ast::Ident {
    fn repr(&self) -> String {
        token::get_ident(*self).to_string()
    }
}

impl Repr for ast::ExplicitSelf_ {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::Visibility {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr for ty::BareFnTy<'tcx> {
    fn repr(&self) -> String {
        format!("BareFnTy {{unsafety: {}, abi: {}, sig: {}}}",
                self.unsafety,
                self.abi.to_string(),
                self.sig.repr())
    }
}


impl<'tcx> Repr for ty::FnSig<'tcx> {
    fn repr(&self) -> String {
        format!("fn{} -> {}", self.inputs.repr(), self.output.repr())
    }
}

impl<'tcx> Repr for ty::FnOutput<'tcx> {
    fn repr(&self) -> String {
        match *self {
            ty::FnConverging(ty) =>
                format!("FnConverging({0})", ty.repr()),
            ty::FnDiverging =>
                "FnDiverging".to_string()
        }
    }
}

impl<'tcx> Repr for ty::MethodCallee<'tcx> {
    fn repr(&self) -> String {
        format!("MethodCallee {{origin: {}, ty: {}, {}}}",
                self.origin.repr(),
                self.ty.repr(),
                self.substs.repr())
    }
}

impl<'tcx> Repr for ty::MethodOrigin<'tcx> {
    fn repr(&self) -> String {
        match self {
            &ty::MethodStatic(def_id) => {
                format!("MethodStatic({})", def_id.repr())
            }
            &ty::MethodStaticClosure(def_id) => {
                format!("MethodStaticClosure({})", def_id.repr())
            }
            &ty::MethodTypeParam(ref p) => {
                p.repr()
            }
            &ty::MethodTraitObject(ref p) => {
                p.repr()
            }
        }
    }
}

impl<'tcx> Repr for ty::MethodParam<'tcx> {
    fn repr(&self) -> String {
        format!("MethodParam({},{})",
                self.trait_ref.repr(),
                self.method_num)
    }
}

impl<'tcx> Repr for ty::MethodObject<'tcx> {
    fn repr(&self) -> String {
        format!("MethodObject({},{},{})",
                self.trait_ref.repr(),
                self.method_num,
                self.vtable_index)
    }
}

impl Repr for ty::BuiltinBound {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl UserString for ty::BuiltinBound {
    fn user_string(&self) -> String {
        match *self {
            ty::BoundSend => "Send".to_string(),
            ty::BoundSized => "Sized".to_string(),
            ty::BoundCopy => "Copy".to_string(),
            ty::BoundSync => "Sync".to_string(),
        }
    }
}

impl Repr for Span {
    fn repr(&self) -> String {
        ty::tls::with(|tcx| tcx.sess.codemap().span_to_string(*self).to_string())
    }
}

impl<A:UserString> UserString for Rc<A> {
    fn user_string(&self) -> String {
        let this: &A = &**self;
        this.user_string()
    }
}

impl<'tcx> UserString for ty::ParamBounds<'tcx> {
    fn user_string(&self) -> String {
        let mut result = Vec::new();
        let s = self.builtin_bounds.user_string();
        if !s.is_empty() {
            result.push(s);
        }
        for n in &self.trait_bounds {
            result.push(n.user_string());
        }
        result.connect(" + ")
    }
}

impl<'tcx> Repr for ty::ExistentialBounds<'tcx> {
    fn repr(&self) -> String {
        let mut res = Vec::new();

        let region_str = self.region_bound.repr();
        if !region_str.is_empty() {
            res.push(region_str);
        }

        for bound in &self.builtin_bounds {
            res.push(bound.repr());
        }

        for projection_bound in &self.projection_bounds {
            res.push(projection_bound.repr());
        }

        res.connect("+")
    }
}

impl UserString for ty::BuiltinBounds {
    fn user_string(&self) -> String {
        self.iter()
            .map(|bb| bb.user_string())
            .collect::<Vec<String>>()
            .connect("+")
            .to_string()
    }
}

// The generic impl doesn't work yet because projections are not
// normalized under HRTB.
/*impl<T> UserString for ty::Binder<T>
    where T: UserString + for<'a> ty::Lift<'a>,
          for<'a> <T as ty::Lift<'a>>::Lifted: UserString + TypeFoldable<'a>
{
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}*/

impl<'tcx> UserString for ty::Binder<ty::TraitRef<'tcx>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::Binder<ty::TraitPredicate<'tcx>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::Binder<ty::EquatePredicate<'tcx>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::Binder<ty::ProjectionPredicate<'tcx>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::Binder<TraitAndProjections<'tcx>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::Binder<ty::OutlivesPredicate<Ty<'tcx>, ty::Region>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl UserString for ty::Binder<ty::OutlivesPredicate<ty::Region, ty::Region>> {
    fn user_string(&self) -> String {
        ty::tls::with(|tcx| in_binder(tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> UserString for ty::TraitRef<'tcx> {
    fn user_string(&self) -> String {
        parameterized(self.substs, self.def_id, &[],
                      |tcx| ty::lookup_trait_def(tcx, self.def_id).generics.clone())
    }
}

impl<'tcx> UserString for ty::TyS<'tcx> {
    fn user_string(&self) -> String {
        fn bare_fn_to_string(opt_def_id: Option<ast::DefId>,
                             unsafety: ast::Unsafety,
                             abi: abi::Abi,
                             ident: Option<ast::Ident>,
                             sig: &ty::PolyFnSig)
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
                s.push_str(&format!("extern {} ", abi.to_string()));
            };

            s.push_str("fn");

            match ident {
                Some(i) => {
                    s.push(' ');
                    s.push_str(&token::get_ident(i));
                }
                _ => { }
            }

            push_sig_to_string(&mut s, '(', ')', sig);

            match opt_def_id {
                Some(def_id) => {
                    s.push_str(" {");
                    let path_str = ty::tls::with(|tcx| ty::item_path_str(tcx, def_id));
                    s.push_str(&path_str[..]);
                    s.push_str("}");
                }
                None => { }
            }

            s
        }

        fn push_sig_to_string(s: &mut String,
                              bra: char,
                              ket: char,
                              sig: &ty::PolyFnSig) {
            s.push(bra);
            let strs = sig.0.inputs
                .iter()
                .map(|a| a.user_string())
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
                        s.push_str(&t.user_string());
                    }
                }
                ty::FnDiverging => {
                    s.push_str(" -> !");
                }
            }
        }

        // pretty print the structural type representation:
        match self.sty {
            TyBool => "bool".to_string(),
            TyChar => "char".to_string(),
            TyInt(t) => ast_util::int_ty_to_string(t, None).to_string(),
            TyUint(t) => ast_util::uint_ty_to_string(t, None).to_string(),
            TyFloat(t) => ast_util::float_ty_to_string(t).to_string(),
            TyBox(typ) => format!("Box<{}>",  typ.user_string()),
            TyRawPtr(ref tm) => {
                format!("*{} {}", match tm.mutbl {
                    ast::MutMutable => "mut",
                    ast::MutImmutable => "const",
                },  tm.ty.user_string())
            }
            TyRef(r, ref tm) => {
                let mut buf = "&".to_owned();
                buf.push_str(&r.user_string());
                if buf.len() > 1 {
                    buf.push_str(" ");
                }
                buf.push_str(&tm.repr());
                buf
            }
            TyTuple(ref elems) => {
                let strs = elems
                    .iter()
                    .map(|elem| elem.user_string())
                    .collect::<Vec<_>>();
                match &strs[..] {
                    [ref string] => format!("({},)", string),
                    strs => format!("({})", strs.connect(", "))
                }
            }
            TyBareFn(opt_def_id, ref f) => {
                bare_fn_to_string(opt_def_id, f.unsafety, f.abi, None, &f.sig)
            }
            TyInfer(infer_ty) => infer_ty.repr(),
            TyError => "[type error]".to_string(),
            TyParam(ref param_ty) => param_ty.user_string(),
            TyEnum(did, substs) | TyStruct(did, substs) => {
                parameterized(substs, did, &[],
                              |tcx| ty::lookup_item_type(tcx, did).generics)
            }
            TyTrait(ref data) => {
                data.user_string()
            }
            ty::TyProjection(ref data) => {
                format!("<{} as {}>::{}",
                        data.trait_ref.self_ty().user_string(),
                        data.trait_ref.user_string(),
                        data.item_name.user_string())
            }
            TyStr => "str".to_string(),
            TyClosure(ref did, substs) => ty::tls::with(|tcx| {
                let closure_tys = tcx.closure_tys.borrow();
                closure_tys.get(did).map(|cty| &cty.sig).and_then(|sig| {
                    tcx.lift(&substs).map(|substs| sig.subst(tcx, substs))
                }).map(|sig| {
                    let mut s = String::new();
                    s.push_str("[closure");
                    push_sig_to_string(&mut s, '(', ')', &sig);
                    if verbose() {
                        s.push_str(&format!(" id={:?}]", did));
                    } else {
                        s.push(']');
                    }
                    s
                }).unwrap_or_else(|| {
                    let id_str = if verbose() {
                        format!(" id={:?}", did)
                    } else {
                        "".to_owned()
                    };

                    if did.krate == ast::LOCAL_CRATE {
                        let span = ty::tls::with(|tcx| tcx.map.span(did.node));
                        format!("[closure {}{}]", span.repr(), id_str)
                    } else {
                        format!("[closure{}]", id_str)
                    }
                })
            }),
            TyArray(t, sz) => {
                format!("[{}; {}]",  t.user_string(), sz)
            }
            TySlice(t) => {
                format!("[{}]",  t.user_string())
            }
        }
    }
}

impl UserString for ast::Ident {
    fn user_string(&self) -> String {
        token::get_name(self.name).to_string()
    }
}

impl Repr for abi::Abi {
    fn repr(&self) -> String {
        self.to_string()
    }
}

impl UserString for abi::Abi {
    fn user_string(&self) -> String {
        self.to_string()
    }
}

impl Repr for ty::UpvarId {
    fn repr(&self) -> String {
        format!("UpvarId({};`{}`;{})",
                self.var_id,
                ty::tls::with(|tcx| ty::local_var_name_str(tcx, self.var_id)),
                self.closure_expr_id)
    }
}

impl Repr for ast::Mutability {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::BorrowKind {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::UpvarBorrow {
    fn repr(&self) -> String {
        format!("UpvarBorrow({}, {})",
                self.kind.repr(),
                self.region.repr())
    }
}

impl Repr for ty::UpvarCapture {
    fn repr(&self) -> String {
        match *self {
            ty::UpvarCapture::ByValue => format!("ByValue"),
            ty::UpvarCapture::ByRef(ref data) => format!("ByRef({})", data.repr()),
        }
    }
}

impl Repr for ty::IntVid {
    fn repr(&self) -> String {
        format!("{:?}", self)
    }
}

impl Repr for ty::FloatVid {
    fn repr(&self) -> String {
        format!("{:?}", self)
    }
}

impl Repr for ty::RegionVid {
    fn repr(&self) -> String {
        format!("{:?}", self)
    }
}

impl Repr for ty::TyVid {
    fn repr(&self) -> String {
        format!("{:?}", self)
    }
}

impl Repr for ty::IntVarValue {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::InferTy {
    fn repr(&self) -> String {
        let print_var_ids = verbose();
        match *self {
            ty::TyVar(ref vid) if print_var_ids => vid.repr(),
            ty::IntVar(ref vid) if print_var_ids => vid.repr(),
            ty::FloatVar(ref vid) if print_var_ids => vid.repr(),
            ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => format!("_"),
            ty::FreshTy(v) => format!("FreshTy({})", v),
            ty::FreshIntTy(v) => format!("FreshIntTy({})", v),
            ty::FreshFloatTy(v) => format!("FreshFloatTy({})", v)
        }
    }
}

impl Repr for ast::IntTy {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::UintTy {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ast::FloatTy {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}

impl Repr for ty::ExplicitSelfCategory {
    fn repr(&self) -> String {
        match *self {
            ty::StaticExplicitSelfCategory => "static",
            ty::ByValueExplicitSelfCategory => "self",
            ty::ByReferenceExplicitSelfCategory(_, ast::MutMutable) => {
                "&mut self"
            }
            ty::ByReferenceExplicitSelfCategory(_, ast::MutImmutable) => "&self",
            ty::ByBoxExplicitSelfCategory => "Box<self>",
        }.to_owned()
    }
}

impl UserString for ParamTy {
    fn user_string(&self) -> String {
        format!("{}", token::get_name(self.name))
    }
}

impl Repr for ParamTy {
    fn repr(&self) -> String {
        let ident = self.user_string();
        format!("{}/{:?}.{}", ident, self.space, self.idx)
    }
}

impl<A:Repr, B:Repr> Repr for (A,B) {
    fn repr(&self) -> String {
        let &(ref a, ref b) = self;
        format!("({},{})", a.repr(), b.repr())
    }
}

impl<T:Repr> Repr for ty::Binder<T> {
    fn repr(&self) -> String {
        format!("Binder({})", self.0.repr())
    }
}

impl<S, K, V> Repr for HashMap<K, V, S>
    where K: Hash + Eq + Repr,
          V: Repr,
          S: HashState,
{
    fn repr(&self) -> String {
        format!("HashMap({})",
                self.iter()
                    .map(|(k,v)| format!("{} => {}", k.repr(), v.repr()))
                    .collect::<Vec<String>>()
                    .connect(", "))
    }
}

impl<'tcx, T, U> Repr for ty::OutlivesPredicate<T,U>
    where T : Repr + TypeFoldable<'tcx>,
          U : Repr + TypeFoldable<'tcx>,
{
    fn repr(&self) -> String {
        format!("OutlivesPredicate({}, {})",
                self.0.repr(),
                self.1.repr())
    }
}

impl<'tcx, T, U> UserString for ty::OutlivesPredicate<T,U>
    where T : UserString + TypeFoldable<'tcx>,
          U : UserString + TypeFoldable<'tcx>,
{
    fn user_string(&self) -> String {
        format!("{} : {}",
                self.0.user_string(),
                self.1.user_string())
    }
}

impl<'tcx> Repr for ty::EquatePredicate<'tcx> {
    fn repr(&self) -> String {
        format!("EquatePredicate({}, {})",
                self.0.repr(),
                self.1.repr())
    }
}

impl<'tcx> UserString for ty::EquatePredicate<'tcx> {
    fn user_string(&self) -> String {
        format!("{} == {}",
                self.0.user_string(),
                self.1.user_string())
    }
}

impl<'tcx> Repr for ty::TraitPredicate<'tcx> {
    fn repr(&self) -> String {
        format!("TraitPredicate({})",
                self.trait_ref.repr())
    }
}

impl<'tcx> UserString for ty::TraitPredicate<'tcx> {
    fn user_string(&self) -> String {
        format!("{} : {}",
                self.trait_ref.self_ty().user_string(),
                self.trait_ref.user_string())
    }
}

impl<'tcx> UserString for ty::ProjectionPredicate<'tcx> {
    fn user_string(&self) -> String {
        format!("{} == {}",
                self.projection_ty.user_string(),
                self.ty.user_string())
    }
}

impl<'tcx> Repr for ty::ProjectionTy<'tcx> {
    fn repr(&self) -> String {
        format!("{}::{}",
                self.trait_ref.repr(),
                self.item_name.repr())
    }
}

impl<'tcx> UserString for ty::ProjectionTy<'tcx> {
    fn user_string(&self) -> String {
        format!("<{} as {}>::{}",
                self.trait_ref.self_ty().user_string(),
                self.trait_ref.user_string(),
                self.item_name.user_string())
    }
}

impl<'tcx> UserString for ty::Predicate<'tcx> {
    fn user_string(&self) -> String {
        match *self {
            ty::Predicate::Trait(ref data) => data.user_string(),
            ty::Predicate::Equate(ref predicate) => predicate.user_string(),
            ty::Predicate::RegionOutlives(ref predicate) => predicate.user_string(),
            ty::Predicate::TypeOutlives(ref predicate) => predicate.user_string(),
            ty::Predicate::Projection(ref predicate) => predicate.user_string(),
        }
    }
}

impl Repr for ast::Unsafety {
    fn repr(&self) -> String {
        format!("{:?}", *self)
    }
}
