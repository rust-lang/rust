// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use hir::map::definitions::DefPathData;
use ty::subst::{self, Subst};
use ty::{BrAnon, BrEnv, BrFresh, BrNamed};
use ty::{TyBool, TyChar, TyAdt};
use ty::{TyError, TyStr, TyArray, TySlice, TyFloat, TyFnDef, TyFnPtr};
use ty::{TyParam, TyRawPtr, TyRef, TyNever, TyTuple};
use ty::{TyClosure, TyProjection, TyAnon};
use ty::{TyBox, TyDynamic, TyInt, TyUint, TyInfer};
use ty::{self, Ty, TyCtxt, TypeFoldable};

use std::cell::Cell;
use std::fmt;
use std::usize;

use syntax::abi::Abi;
use syntax::ast::CRATE_NODE_ID;
use syntax::symbol::Symbol;
use hir;

pub fn verbose() -> bool {
    ty::tls::with(|tcx| tcx.sess.verbose())
}

fn fn_sig(f: &mut fmt::Formatter,
          inputs: &[Ty],
          variadic: bool,
          output: Ty)
          -> fmt::Result {
    write!(f, "(")?;
    let mut inputs = inputs.iter();
    if let Some(&ty) = inputs.next() {
        write!(f, "{}", ty)?;
        for &ty in inputs {
            write!(f, ", {}", ty)?;
        }
        if variadic {
            write!(f, ", ...")?;
        }
    }
    write!(f, ")")?;
    if !output.is_nil() {
        write!(f, " -> {}", output)?;
    }

    Ok(())
}

pub fn parameterized(f: &mut fmt::Formatter,
                     substs: &subst::Substs,
                     mut did: DefId,
                     projections: &[ty::ProjectionPredicate])
                     -> fmt::Result {
    let key = ty::tls::with(|tcx| tcx.def_key(did));
    let mut item_name = if let Some(name) = key.disambiguated_data.data.get_opt_name() {
        Some(name)
    } else {
        did.index = key.parent.unwrap_or_else(
            || bug!("finding type for {:?}, encountered def-id {:?} with no parent",
                    did, did));
        parameterized(f, substs, did, projections)?;
        return write!(f, "::{}", key.disambiguated_data.data.as_interned_str());
    };

    let mut verbose = false;
    let mut num_supplied_defaults = 0;
    let mut has_self = false;
    let mut num_regions = 0;
    let mut num_types = 0;
    let mut is_value_path = false;
    let fn_trait_kind = ty::tls::with(|tcx| {
        // Unfortunately, some kinds of items (e.g., closures) don't have
        // generics. So walk back up the find the closest parent that DOES
        // have them.
        let mut item_def_id = did;
        loop {
            let key = tcx.def_key(item_def_id);
            match key.disambiguated_data.data {
                DefPathData::TypeNs(_) => {
                    break;
                }
                DefPathData::ValueNs(_) | DefPathData::EnumVariant(_) => {
                    is_value_path = true;
                    break;
                }
                _ => {
                    // if we're making a symbol for something, there ought
                    // to be a value or type-def or something in there
                    // *somewhere*
                    item_def_id.index = key.parent.unwrap_or_else(|| {
                        bug!("finding type for {:?}, encountered def-id {:?} with no \
                             parent", did, item_def_id);
                    });
                }
            }
        }
        let mut generics = tcx.item_generics(item_def_id);
        let mut path_def_id = did;
        verbose = tcx.sess.verbose();
        has_self = generics.has_self;

        let mut child_types = 0;
        if let Some(def_id) = generics.parent {
            // Methods.
            assert!(is_value_path);
            child_types = generics.types.len();
            generics = tcx.item_generics(def_id);
            num_regions = generics.regions.len();
            num_types = generics.types.len();

            if has_self {
                write!(f, "<{} as ", substs.type_at(0))?;
            }

            path_def_id = def_id;
        } else {
            item_name = None;

            if is_value_path {
                // Functions.
                assert_eq!(has_self, false);
            } else {
                // Types and traits.
                num_regions = generics.regions.len();
                num_types = generics.types.len();
            }
        }

        if !verbose {
            if generics.types.last().map_or(false, |def| def.default.is_some()) {
                if let Some(substs) = tcx.lift(&substs) {
                    let tps = substs.types().rev().skip(child_types);
                    for (def, actual) in generics.types.iter().rev().zip(tps) {
                        if def.default.subst(tcx, substs) != Some(actual) {
                            break;
                        }
                        num_supplied_defaults += 1;
                    }
                }
            }
        }

        write!(f, "{}", tcx.item_path_str(path_def_id))?;
        Ok(tcx.lang_items.fn_trait_kind(path_def_id))
    })?;

    if !verbose && fn_trait_kind.is_some() && projections.len() == 1 {
        let projection_ty = projections[0].ty;
        if let TyTuple(ref args) = substs.type_at(1).sty {
            return fn_sig(f, args, false, projection_ty);
        }
    }

    let empty = Cell::new(true);
    let start_or_continue = |f: &mut fmt::Formatter, start: &str, cont: &str| {
        if empty.get() {
            empty.set(false);
            write!(f, "{}", start)
        } else {
            write!(f, "{}", cont)
        }
    };

    let print_regions = |f: &mut fmt::Formatter, start: &str, skip, count| {
        // Don't print any regions if they're all erased.
        let regions = || substs.regions().skip(skip).take(count);
        if regions().all(|r: &ty::Region| *r == ty::ReErased) {
            return Ok(());
        }

        for region in regions() {
            let region: &ty::Region = region;
            start_or_continue(f, start, ", ")?;
            if verbose {
                write!(f, "{:?}", region)?;
            } else {
                let s = region.to_string();
                if s.is_empty() {
                    // This happens when the value of the region
                    // parameter is not easily serialized. This may be
                    // because the user omitted it in the first place,
                    // or because it refers to some block in the code,
                    // etc. I'm not sure how best to serialize this.
                    write!(f, "'_")?;
                } else {
                    write!(f, "{}", s)?;
                }
            }
        }

        Ok(())
    };

    print_regions(f, "<", 0, num_regions)?;

    let tps = substs.types().take(num_types - num_supplied_defaults)
                            .skip(has_self as usize);

    for ty in tps {
        start_or_continue(f, "<", ", ")?;
        write!(f, "{}", ty)?;
    }

    for projection in projections {
        start_or_continue(f, "<", ", ")?;
        write!(f, "{}={}",
               projection.projection_ty.item_name,
               projection.ty)?;
    }

    start_or_continue(f, "", ">")?;

    // For values, also print their name and type parameters.
    if is_value_path {
        empty.set(true);

        if has_self {
            write!(f, ">")?;
        }

        if let Some(item_name) = item_name {
            write!(f, "::{}", item_name)?;
        }

        print_regions(f, "::<", num_regions, usize::MAX)?;

        // FIXME: consider being smart with defaults here too
        for ty in substs.types().skip(num_types) {
            start_or_continue(f, "::<", ", ")?;
            write!(f, "{}", ty)?;
        }

        start_or_continue(f, "", ">")?;
    }

    Ok(())
}

fn in_binder<'a, 'gcx, 'tcx, T, U>(f: &mut fmt::Formatter,
                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                   original: &ty::Binder<T>,
                                   lifted: Option<ty::Binder<U>>) -> fmt::Result
    where T: fmt::Display, U: fmt::Display + TypeFoldable<'tcx>
{
    // Replace any anonymous late-bound regions with named
    // variants, using gensym'd identifiers, so that we can
    // clearly differentiate between named and unnamed regions in
    // the output. We'll probably want to tweak this over time to
    // decide just how much information to give.
    let value = if let Some(v) = lifted {
        v
    } else {
        return write!(f, "{}", original.0);
    };

    let mut empty = true;
    let mut start_or_continue = |f: &mut fmt::Formatter, start: &str, cont: &str| {
        if empty {
            empty = false;
            write!(f, "{}", start)
        } else {
            write!(f, "{}", cont)
        }
    };

    let new_value = tcx.replace_late_bound_regions(&value, |br| {
        let _ = start_or_continue(f, "for<", ", ");
        let br = match br {
            ty::BrNamed(_, name, _) => {
                let _ = write!(f, "{}", name);
                br
            }
            ty::BrAnon(_) |
            ty::BrFresh(_) |
            ty::BrEnv => {
                let name = Symbol::intern("'r");
                let _ = write!(f, "{}", name);
                ty::BrNamed(tcx.map.local_def_id(CRATE_NODE_ID),
                            name,
                            ty::Issue32330::WontChange)
            }
        };
        tcx.mk_region(ty::ReLateBound(ty::DebruijnIndex::new(1), br))
    }).0;

    start_or_continue(f, "", "> ")?;
    write!(f, "{}", new_value)
}

impl<'tcx> fmt::Display for &'tcx ty::Slice<ty::ExistentialPredicate<'tcx>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Generate the main trait ref, including associated types.
        ty::tls::with(|tcx| {
            // Use a type that can't appear in defaults of type parameters.
            let dummy_self = tcx.mk_infer(ty::FreshTy(0));

            if let Some(p) = self.principal() {
                let principal = tcx.lift(&p).expect("could not lift TraitRef for printing")
                    .with_self_ty(tcx, dummy_self);
                let projections = self.projection_bounds().map(|p| {
                    tcx.lift(&p)
                        .expect("could not lift projection for printing")
                        .with_self_ty(tcx, dummy_self)
                }).collect::<Vec<_>>();
                parameterized(f, principal.substs, principal.def_id, &projections)?;
            }

            // Builtin bounds.
            for did in self.auto_traits() {
                write!(f, " + {}", tcx.item_path_str(did))?;
            }

            Ok(())
        })?;

        Ok(())
    }
}

impl<'tcx> fmt::Debug for ty::TypeParameterDef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeParameterDef({}, {:?}, {})",
               self.name,
               self.def_id,
               self.index)
    }
}

impl<'tcx> fmt::Debug for ty::RegionParameterDef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionParameterDef({}, {:?}, {}, {:?})",
               self.name,
               self.def_id,
               self.index,
               self.bounds)
    }
}

impl<'tcx> fmt::Debug for ty::TyS<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", *self)
    }
}

impl<'tcx> fmt::Display for ty::TypeAndMut<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}",
               if self.mutbl == hir::MutMutable { "mut " } else { "" },
               self.ty)
    }
}

impl<'tcx> fmt::Debug for ty::ItemSubsts<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ItemSubsts({:?})", self.substs)
    }
}

impl<'tcx> fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // when printing out the debug representation, we don't need
        // to enumerate the `for<...>` etc because the debruijn index
        // tells you everything you need to know.
        write!(f, "<{:?} as {}>", self.self_ty(), *self)
    }
}

impl<'tcx> fmt::Debug for ty::ExistentialTraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| {
            let dummy_self = tcx.mk_infer(ty::FreshTy(0));

            let trait_ref = tcx.lift(&ty::Binder(*self))
                               .expect("could not lift TraitRef for printing")
                               .with_self_ty(tcx, dummy_self).0;
            parameterized(f, trait_ref.substs, trait_ref.def_id, &[])
        })
    }
}

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| {
            write!(f, "{}", tcx.item_path_str(self.def_id))
        })
    }
}

impl fmt::Debug for ty::AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| {
            write!(f, "{}", tcx.item_path_str(self.did))
        })
    }
}

impl<'tcx> fmt::Debug for ty::adjustment::Adjustment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} -> {}", self.kind, self.target)
    }
}

impl<'tcx> fmt::Debug for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::Predicate::Trait(ref a) => write!(f, "{:?}", a),
            ty::Predicate::Equate(ref pair) => write!(f, "{:?}", pair),
            ty::Predicate::RegionOutlives(ref pair) => write!(f, "{:?}", pair),
            ty::Predicate::TypeOutlives(ref pair) => write!(f, "{:?}", pair),
            ty::Predicate::Projection(ref pair) => write!(f, "{:?}", pair),
            ty::Predicate::WellFormed(ty) => write!(f, "WF({:?})", ty),
            ty::Predicate::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({:?})", trait_def_id)
            }
            ty::Predicate::ClosureKind(closure_def_id, kind) => {
                write!(f, "ClosureKind({:?}, {:?})", closure_def_id, kind)
            }
        }
    }
}

impl fmt::Display for ty::BoundRegion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if verbose() {
            return write!(f, "{:?}", *self);
        }

        match *self {
            BrNamed(_, name, _) => write!(f, "{}", name),
            BrAnon(_) | BrFresh(_) | BrEnv => Ok(())
        }
    }
}

impl fmt::Debug for ty::BoundRegion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BrAnon(n) => write!(f, "BrAnon({:?})", n),
            BrFresh(n) => write!(f, "BrFresh({:?})", n),
            BrNamed(did, name, issue32330) => {
                write!(f, "BrNamed({:?}:{:?}, {:?}, {:?})",
                       did.krate, did.index, name, issue32330)
            }
            BrEnv => "BrEnv".fmt(f),
        }
    }
}

impl fmt::Debug for ty::Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::ReEarlyBound(ref data) => {
                write!(f, "ReEarlyBound({}, {})",
                       data.index,
                       data.name)
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                write!(f, "ReLateBound({:?}, {:?})",
                       binder_id,
                       bound_region)
            }

            ty::ReFree(ref fr) => write!(f, "{:?}", fr),

            ty::ReScope(id) => {
                write!(f, "ReScope({:?})", id)
            }

            ty::ReStatic => write!(f, "ReStatic"),

            ty::ReVar(ref vid) => {
                write!(f, "{:?}", vid)
            }

            ty::ReSkolemized(id, ref bound_region) => {
                write!(f, "ReSkolemized({}, {:?})", id.index, bound_region)
            }

            ty::ReEmpty => write!(f, "ReEmpty"),

            ty::ReErased => write!(f, "ReErased")
        }
    }
}

impl<'tcx> fmt::Debug for ty::ClosureTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ClosureTy({},{:?},{})",
               self.unsafety,
               self.sig,
               self.abi)
    }
}

impl<'tcx> fmt::Debug for ty::ClosureUpvar<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ClosureUpvar({:?},{:?})",
               self.def,
               self.ty)
    }
}

impl<'tcx> fmt::Debug for ty::ParameterEnvironment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParameterEnvironment(\
            free_substs={:?}, \
            implicit_region_bound={:?}, \
            caller_bounds={:?})",
            self.free_substs,
            self.implicit_region_bound,
            self.caller_bounds)
    }
}

impl<'tcx> fmt::Debug for ty::ObjectLifetimeDefault<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::ObjectLifetimeDefault::Ambiguous => write!(f, "Ambiguous"),
            ty::ObjectLifetimeDefault::BaseDefault => write!(f, "BaseDefault"),
            ty::ObjectLifetimeDefault::Specific(ref r) => write!(f, "{:?}", r),
        }
    }
}

impl fmt::Display for ty::Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if verbose() {
            return write!(f, "{:?}", *self);
        }

        // These printouts are concise.  They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string.  Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *self {
            ty::ReEarlyBound(ref data) => {
                write!(f, "{}", data.name)
            }
            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) |
            ty::ReSkolemized(_, br) => {
                write!(f, "{}", br)
            }
            ty::ReScope(_) |
            ty::ReVar(_) |
            ty::ReErased => Ok(()),
            ty::ReStatic => write!(f, "'static"),
            ty::ReEmpty => write!(f, "'<empty>"),
        }
    }
}

impl fmt::Debug for ty::FreeRegion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ReFree({:?}, {:?})",
               self.scope, self.bound_region)
    }
}

impl fmt::Debug for ty::Variance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            ty::Covariant => "+",
            ty::Contravariant => "-",
            ty::Invariant => "o",
            ty::Bivariant => "*",
        })
    }
}

impl<'tcx> fmt::Debug for ty::GenericPredicates<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GenericPredicates({:?})", self.predicates)
    }
}

impl<'tcx> fmt::Debug for ty::InstantiatedPredicates<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InstantiatedPredicates({:?})",
               self.predicates)
    }
}

impl<'tcx> fmt::Display for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "fn")?;
        fn_sig(f, self.inputs(), self.variadic, self.output())
    }
}

impl fmt::Debug for ty::TyVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_#{}t", self.index)
    }
}

impl fmt::Debug for ty::IntVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_#{}i", self.index)
    }
}

impl fmt::Debug for ty::FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_#{}f", self.index)
    }
}

impl fmt::Debug for ty::RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "'_#{}r", self.index)
    }
}

impl<'tcx> fmt::Debug for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({:?}; variadic: {})->{:?}", self.inputs(), self.variadic, self.output())
    }
}

impl fmt::Debug for ty::InferTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::TyVar(ref v) => v.fmt(f),
            ty::IntVar(ref v) => v.fmt(f),
            ty::FloatVar(ref v) => v.fmt(f),
            ty::FreshTy(v) => write!(f, "FreshTy({:?})", v),
            ty::FreshIntTy(v) => write!(f, "FreshIntTy({:?})", v),
            ty::FreshFloatTy(v) => write!(f, "FreshFloatTy({:?})", v)
        }
    }
}

impl fmt::Debug for ty::IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::IntType(ref v) => v.fmt(f),
            ty::UintType(ref v) => v.fmt(f),
        }
    }
}

// The generic impl doesn't work yet because projections are not
// normalized under HRTB.
/*impl<T> fmt::Display for ty::Binder<T>
    where T: fmt::Display + for<'a> ty::Lift<'a>,
          for<'a> <T as ty::Lift<'a>>::Lifted: fmt::Display + TypeFoldable<'a>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}*/

impl<'tcx> fmt::Display for ty::Binder<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::TraitRef<'tcx>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::TraitPredicate<'tcx>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::EquatePredicate<'tcx>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::ProjectionPredicate<'tcx>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::OutlivesPredicate<Ty<'tcx>, &'tcx ty::Region>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::Binder<ty::OutlivesPredicate<&'tcx ty::Region,
                                                             &'tcx ty::Region>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ty::tls::with(|tcx| in_binder(f, tcx, self, tcx.lift(self)))
    }
}

impl<'tcx> fmt::Display for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        parameterized(f, self.substs, self.def_id, &[])
    }
}

impl<'tcx> fmt::Display for ty::TypeVariants<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TyBool => write!(f, "bool"),
            TyChar => write!(f, "char"),
            TyInt(t) => write!(f, "{}", t.ty_to_string()),
            TyUint(t) => write!(f, "{}", t.ty_to_string()),
            TyFloat(t) => write!(f, "{}", t.ty_to_string()),
            TyBox(typ) => write!(f, "Box<{}>",  typ),
            TyRawPtr(ref tm) => {
                write!(f, "*{} {}", match tm.mutbl {
                    hir::MutMutable => "mut",
                    hir::MutImmutable => "const",
                },  tm.ty)
            }
            TyRef(r, ref tm) => {
                write!(f, "&")?;
                let s = r.to_string();
                write!(f, "{}", s)?;
                if !s.is_empty() {
                    write!(f, " ")?;
                }
                write!(f, "{}", tm)
            }
            TyNever => write!(f, "!"),
            TyTuple(ref tys) => {
                write!(f, "(")?;
                let mut tys = tys.iter();
                if let Some(&ty) = tys.next() {
                    write!(f, "{},", ty)?;
                    if let Some(&ty) = tys.next() {
                        write!(f, " {}", ty)?;
                        for &ty in tys {
                            write!(f, ", {}", ty)?;
                        }
                    }
                }
                write!(f, ")")
            }
            TyFnDef(def_id, substs, ref bare_fn) => {
                if bare_fn.unsafety == hir::Unsafety::Unsafe {
                    write!(f, "unsafe ")?;
                }

                if bare_fn.abi != Abi::Rust {
                    write!(f, "extern {} ", bare_fn.abi)?;
                }

                write!(f, "{} {{", bare_fn.sig.0)?;
                parameterized(f, substs, def_id, &[])?;
                write!(f, "}}")
            }
            TyFnPtr(ref bare_fn) => {
                if bare_fn.unsafety == hir::Unsafety::Unsafe {
                    write!(f, "unsafe ")?;
                }

                if bare_fn.abi != Abi::Rust {
                    write!(f, "extern {} ", bare_fn.abi)?;
                }

                write!(f, "{}", bare_fn.sig.0)
            }
            TyInfer(infer_ty) => write!(f, "{}", infer_ty),
            TyError => write!(f, "[type error]"),
            TyParam(ref param_ty) => write!(f, "{}", param_ty),
            TyAdt(def, substs) => {
                ty::tls::with(|tcx| {
                    if def.did.is_local() &&
                          !tcx.item_types.borrow().contains_key(&def.did) {
                        write!(f, "{}<..>", tcx.item_path_str(def.did))
                    } else {
                        parameterized(f, substs, def.did, &[])
                    }
                })
            }
            TyDynamic(data, r) => {
                write!(f, "{}", data)?;
                let r = r.to_string();
                if !r.is_empty() {
                    write!(f, " + {}", r)
                } else {
                    Ok(())
                }
            }
            TyProjection(ref data) => write!(f, "{}", data),
            TyAnon(def_id, substs) => {
                ty::tls::with(|tcx| {
                    // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                    // by looking up the projections associated with the def_id.
                    let item_predicates = tcx.item_predicates(def_id);
                    let substs = tcx.lift(&substs).unwrap_or_else(|| {
                        tcx.intern_substs(&[])
                    });
                    let bounds = item_predicates.instantiate(tcx, substs);

                    let mut first = true;
                    let mut is_sized = false;
                    write!(f, "impl")?;
                    for predicate in bounds.predicates {
                        if let Some(trait_ref) = predicate.to_opt_poly_trait_ref() {
                            // Don't print +Sized, but rather +?Sized if absent.
                            if Some(trait_ref.def_id()) == tcx.lang_items.sized_trait() {
                                is_sized = true;
                                continue;
                            }

                            write!(f, "{}{}", if first { " " } else { "+" }, trait_ref)?;
                            first = false;
                        }
                    }
                    if !is_sized {
                        write!(f, "{}?Sized", if first { " " } else { "+" })?;
                    }
                    Ok(())
                })
            }
            TyStr => write!(f, "str"),
            TyClosure(did, substs) => ty::tls::with(|tcx| {
                let upvar_tys = substs.upvar_tys(did, tcx);
                write!(f, "[closure")?;

                if let Some(node_id) = tcx.map.as_local_node_id(did) {
                    write!(f, "@{:?}", tcx.map.span(node_id))?;
                    let mut sep = " ";
                    tcx.with_freevars(node_id, |freevars| {
                        for (freevar, upvar_ty) in freevars.iter().zip(upvar_tys) {
                            let def_id = freevar.def.def_id();
                            let node_id = tcx.map.as_local_node_id(def_id).unwrap();
                            write!(f,
                                        "{}{}:{}",
                                        sep,
                                        tcx.local_var_name_str(node_id),
                                        upvar_ty)?;
                            sep = ", ";
                        }
                        Ok(())
                    })?
                } else {
                    // cross-crate closure types should only be
                    // visible in trans bug reports, I imagine.
                    write!(f, "@{:?}", did)?;
                    let mut sep = " ";
                    for (index, upvar_ty) in upvar_tys.enumerate() {
                        write!(f, "{}{}:{}", sep, index, upvar_ty)?;
                        sep = ", ";
                    }
                }

                write!(f, "]")
            }),
            TyArray(ty, sz) => write!(f, "[{}; {}]",  ty, sz),
            TySlice(ty) => write!(f, "[{}]",  ty)
        }
    }
}

impl<'tcx> fmt::Display for ty::TyS<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.sty)
    }
}

impl fmt::Debug for ty::UpvarId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UpvarId({};`{}`;{})",
               self.var_id,
               ty::tls::with(|tcx| tcx.local_var_name_str(self.var_id)),
               self.closure_expr_id)
    }
}

impl<'tcx> fmt::Debug for ty::UpvarBorrow<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UpvarBorrow({:?}, {:?})",
               self.kind, self.region)
    }
}

impl fmt::Display for ty::InferTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_var_ids = verbose();
        match *self {
            ty::TyVar(ref vid) if print_var_ids => write!(f, "{:?}", vid),
            ty::IntVar(ref vid) if print_var_ids => write!(f, "{:?}", vid),
            ty::FloatVar(ref vid) if print_var_ids => write!(f, "{:?}", vid),
            ty::TyVar(_) => write!(f, "_"),
            ty::IntVar(_) => write!(f, "{}", "{integer}"),
            ty::FloatVar(_) => write!(f, "{}", "{float}"),
            ty::FreshTy(v) => write!(f, "FreshTy({})", v),
            ty::FreshIntTy(v) => write!(f, "FreshIntTy({})", v),
            ty::FreshFloatTy(v) => write!(f, "FreshFloatTy({})", v)
        }
    }
}

impl fmt::Display for ty::ParamTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Debug for ty::ParamTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/#{}", self, self.idx)
    }
}

impl<'tcx, T, U> fmt::Display for ty::OutlivesPredicate<T,U>
    where T: fmt::Display, U: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} : {}", self.0, self.1)
    }
}

impl<'tcx> fmt::Display for ty::EquatePredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} == {}", self.0, self.1)
    }
}

impl<'tcx> fmt::Debug for ty::TraitPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TraitPredicate({:?})",
               self.trait_ref)
    }
}

impl<'tcx> fmt::Display for ty::TraitPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.trait_ref.self_ty(), self.trait_ref)
    }
}

impl<'tcx> fmt::Debug for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})",
               self.projection_ty,
               self.ty)
    }
}

impl<'tcx> fmt::Display for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} == {}",
               self.projection_ty,
               self.ty)
    }
}

impl<'tcx> fmt::Display for ty::ProjectionTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}::{}",
               self.trait_ref,
               self.item_name)
    }
}

impl fmt::Display for ty::ClosureKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::ClosureKind::Fn => write!(f, "Fn"),
            ty::ClosureKind::FnMut => write!(f, "FnMut"),
            ty::ClosureKind::FnOnce => write!(f, "FnOnce"),
        }
    }
}

impl<'tcx> fmt::Display for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ty::Predicate::Trait(ref data) => write!(f, "{}", data),
            ty::Predicate::Equate(ref predicate) => write!(f, "{}", predicate),
            ty::Predicate::RegionOutlives(ref predicate) => write!(f, "{}", predicate),
            ty::Predicate::TypeOutlives(ref predicate) => write!(f, "{}", predicate),
            ty::Predicate::Projection(ref predicate) => write!(f, "{}", predicate),
            ty::Predicate::WellFormed(ty) => write!(f, "{} well-formed", ty),
            ty::Predicate::ObjectSafe(trait_def_id) =>
                ty::tls::with(|tcx| {
                    write!(f, "the trait `{}` is object-safe", tcx.item_path_str(trait_def_id))
                }),
            ty::Predicate::ClosureKind(closure_def_id, kind) =>
                ty::tls::with(|tcx| {
                    write!(f, "the closure `{}` implements the trait `{}`",
                           tcx.item_path_str(closure_def_id), kind)
                }),
        }
    }
}
