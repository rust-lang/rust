// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type substitutions.

use middle::cstore;
use hir::def_id::DefId;
use ty::{self, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder};

use serialize::{Encodable, Encoder, Decodable, Decoder};
use syntax_pos::{Span, DUMMY_SP};

///////////////////////////////////////////////////////////////////////////

/// A substitution mapping type/region parameters to new values.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Substs<'tcx> {
    pub types: Vec<Ty<'tcx>>,
    pub regions: Vec<ty::Region>,
}

impl<'a, 'gcx, 'tcx> Substs<'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               t: Vec<Ty<'tcx>>,
               r: Vec<ty::Region>)
               -> &'tcx Substs<'tcx>
    {
        tcx.mk_substs(Substs { types: t, regions: r })
    }

    pub fn new_trait(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                     mut t: Vec<Ty<'tcx>>,
                     r: Vec<ty::Region>,
                     s: Ty<'tcx>)
                    -> &'tcx Substs<'tcx>
    {
        t.insert(0, s);
        Substs::new(tcx, t, r)
    }

    pub fn empty(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> &'tcx Substs<'tcx> {
        Substs::new(tcx, vec![], vec![])
    }

    /// Creates a Substs for generic parameter definitions,
    /// by calling closures to obtain each region and type.
    /// The closures get to observe the Substs as they're
    /// being built, which can be used to correctly
    /// substitute defaults of type parameters.
    pub fn for_item<FR, FT>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                            def_id: DefId,
                            mut mk_region: FR,
                            mut mk_type: FT)
                            -> &'tcx Substs<'tcx>
    where FR: FnMut(&ty::RegionParameterDef, &Substs<'tcx>) -> ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &Substs<'tcx>) -> Ty<'tcx> {
        let defs = tcx.lookup_generics(def_id);
        let num_regions = defs.parent_regions as usize + defs.regions.len();
        let num_types = defs.parent_types as usize + defs.types.len();
        let mut substs = Substs {
            regions: Vec::with_capacity(num_regions),
            types: Vec::with_capacity(num_types)
        };

        substs.fill_item(tcx, defs, &mut mk_region, &mut mk_type);

        Substs::new(tcx, substs.types, substs.regions)
    }

    fn fill_item<FR, FT>(&mut self,
                         tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         defs: &ty::Generics<'tcx>,
                         mk_region: &mut FR,
                         mk_type: &mut FT)
    where FR: FnMut(&ty::RegionParameterDef, &Substs<'tcx>) -> ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &Substs<'tcx>) -> Ty<'tcx> {
        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.lookup_generics(def_id);
            self.fill_item(tcx, parent_defs, mk_region, mk_type);
        }

        for def in &defs.regions {
            let region = mk_region(def, self);
            assert_eq!(def.index as usize, self.regions.len());
            self.regions.push(region);
        }

        for def in &defs.types {
            let ty = mk_type(def, self);
            assert_eq!(def.index as usize, self.types.len());
            self.types.push(ty);
        }
    }

    pub fn is_noop(&self) -> bool {
        self.regions.is_empty() && self.types.is_empty()
    }

    pub fn type_for_def(&self, ty_param_def: &ty::TypeParameterDef) -> Ty<'tcx> {
        self.types[ty_param_def.index as usize]
    }

    pub fn region_for_def(&self, def: &ty::RegionParameterDef) -> ty::Region {
        self.regions[def.index as usize]
    }

    /// Transform from substitutions for a child of `source_ancestor`
    /// (e.g. a trait or impl) to substitutions for the same child
    /// in a different item, with `target_substs` as the base for
    /// the target impl/trait, with the source child-specific
    /// parameters (e.g. method parameters) on top of that base.
    pub fn rebase_onto(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                       source_ancestor: DefId,
                       target_substs: &Substs<'tcx>)
                       -> &'tcx Substs<'tcx> {
        let defs = tcx.lookup_generics(source_ancestor);
        let regions = target_substs.regions.iter()
            .chain(&self.regions[defs.regions.len()..]).cloned().collect();
        let types = target_substs.types.iter()
            .chain(&self.types[defs.types.len()..]).cloned().collect();
        Substs::new(tcx, types, regions)
    }
}

impl<'tcx> Encodable for &'tcx Substs<'tcx> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        cstore::tls::with_encoding_context(s, |ecx, rbml_w| {
            ecx.encode_substs(rbml_w, self);
            Ok(())
        })
    }
}

impl<'tcx> Decodable for &'tcx Substs<'tcx> {
    fn decode<D: Decoder>(d: &mut D) -> Result<&'tcx Substs<'tcx>, D::Error> {
        let substs = cstore::tls::with_decoding_context(d, |dcx, rbml_r| {
            dcx.decode_substs(rbml_r)
        });

        Ok(substs)
    }
}


///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`. Or use `foo.subst_spanned(tcx, substs, Some(span))` when
// there is more information available (for better errors).

pub trait Subst<'tcx> : Sized {
    fn subst<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      substs: &Substs<'tcx>) -> Self {
        self.subst_spanned(tcx, substs, None)
    }

    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &Substs<'tcx>,
                               span: Option<Span>)
                               -> Self;
}

impl<'tcx, T:TypeFoldable<'tcx>> Subst<'tcx> for T {
    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &Substs<'tcx>,
                               span: Option<Span>)
                               -> T
    {
        let mut folder = SubstFolder { tcx: tcx,
                                       substs: substs,
                                       span: span,
                                       root_ty: None,
                                       ty_stack_depth: 0,
                                       region_binders_passed: 0 };
        (*self).fold_with(&mut folder)
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual substitution engine itself is a type folder.

struct SubstFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    substs: &'a Substs<'tcx>,

    // The location for which the substitution is performed, if available.
    span: Option<Span>,

    // The root type that is being substituted, if available.
    root_ty: Option<Ty<'tcx>>,

    // Depth of type stack
    ty_stack_depth: usize,

    // Number of region binders we have passed through while doing the substitution
    region_binders_passed: u32,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for SubstFolder<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T> {
        self.region_binders_passed += 1;
        let t = t.super_fold_with(self);
        self.region_binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match r {
            ty::ReEarlyBound(data) => {
                match self.substs.regions.get(data.index as usize) {
                    Some(&r) => {
                        self.shift_region_through_binders(r)
                    }
                    None => {
                        let span = self.span.unwrap_or(DUMMY_SP);
                        span_bug!(
                            span,
                            "Region parameter out of range \
                             when substituting in region {} (root type={:?}) \
                             (index={})",
                            data.name,
                            self.root_ty,
                            data.index);
                    }
                }
            }
            _ => r
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_subst() {
            return t;
        }

        // track the root type we were asked to substitute
        let depth = self.ty_stack_depth;
        if depth == 0 {
            self.root_ty = Some(t);
        }
        self.ty_stack_depth += 1;

        let t1 = match t.sty {
            ty::TyParam(p) => {
                self.ty_for_param(p, t)
            }
            _ => {
                t.super_fold_with(self)
            }
        };

        assert_eq!(depth + 1, self.ty_stack_depth);
        self.ty_stack_depth -= 1;
        if depth == 0 {
            self.root_ty = None;
        }

        return t1;
    }
}

impl<'a, 'gcx, 'tcx> SubstFolder<'a, 'gcx, 'tcx> {
    fn ty_for_param(&self, p: ty::ParamTy, source_ty: Ty<'tcx>) -> Ty<'tcx> {
        // Look up the type in the substitutions. It really should be in there.
        let opt_ty = self.substs.types.get(p.idx as usize);
        let ty = match opt_ty {
            Some(t) => *t,
            None => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "Type parameter `{:?}` ({:?}/{}) out of range \
                         when substituting (root type={:?}) substs={:?}",
                    p,
                    source_ty,
                    p.idx,
                    self.root_ty,
                    self.substs);
            }
        };

        self.shift_regions_through_binders(ty)
    }

    /// It is sometimes necessary to adjust the debruijn indices during substitution. This occurs
    /// when we are substituting a type with escaping regions into a context where we have passed
    /// through region binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a int>)
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    ///
    ///     for<'a> fn(fn(&'a int))
    ///             ^~ ^~ ^~~
    ///             |  |  |
    ///             |  |  DebruijnIndex of 2
    ///             Binders
    ///
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count Debruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a int` will have a
    /// debruijn index of 1. It's only during the substitution that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a int>)
    /// ```
    ///
    /// Here the final type will be:
    ///
    ///     for<'a> fn((&'a int, fn(&'a int)))
    ///                 ^~~         ^~~
    ///                 |           |
    ///          DebruijnIndex of 1 |
    ///                      DebruijnIndex of 2
    ///
    /// As indicated in the diagram, here the same type `&'a int` is substituted once, but in the
    /// first case we do not increase the Debruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    fn shift_regions_through_binders(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("shift_regions(ty={:?}, region_binders_passed={:?}, has_escaping_regions={:?})",
               ty, self.region_binders_passed, ty.has_escaping_regions());

        if self.region_binders_passed == 0 || !ty.has_escaping_regions() {
            return ty;
        }

        let result = ty::fold::shift_regions(self.tcx(), self.region_binders_passed, &ty);
        debug!("shift_regions: shifted result = {:?}", result);

        result
    }

    fn shift_region_through_binders(&self, region: ty::Region) -> ty::Region {
        ty::fold::shift_region(region, self.region_binders_passed)
    }
}

// Helper methods that modify substitutions.

impl<'a, 'gcx, 'tcx> ty::TraitRef<'tcx> {
    pub fn from_method(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                       trait_id: DefId,
                       substs: &Substs<'tcx>)
                       -> ty::TraitRef<'tcx> {
        let defs = tcx.lookup_generics(trait_id);
        let regions = substs.regions[..defs.regions.len()].to_vec();
        let types = substs.types[..defs.types.len()].to_vec();

        ty::TraitRef {
            def_id: trait_id,
            substs: Substs::new(tcx, types, regions)
        }
    }
}

impl<'a, 'gcx, 'tcx> ty::ExistentialTraitRef<'tcx> {
    pub fn erase_self_ty(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         trait_ref: ty::TraitRef<'tcx>)
                         -> ty::ExistentialTraitRef<'tcx> {
        let Substs { mut types, regions } = trait_ref.substs.clone();

        types.remove(0);

        ty::ExistentialTraitRef {
            def_id: trait_ref.def_id,
            substs: Substs::new(tcx, types, regions)
        }
    }
}

impl<'a, 'gcx, 'tcx> ty::PolyExistentialTraitRef<'tcx> {
    /// Object types don't have a self-type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self-type. A common choice is `mk_err()`
    /// or some skolemized type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        self_ty: Ty<'tcx>)
                        -> ty::PolyTraitRef<'tcx>  {
        // otherwise the escaping regions would be captured by the binder
        assert!(!self_ty.has_escaping_regions());

        self.map_bound(|trait_ref| {
            let Substs { mut types, regions } = trait_ref.substs.clone();

            types.insert(0, self_ty);

            ty::TraitRef {
                def_id: trait_ref.def_id,
                substs: Substs::new(tcx, types, regions)
            }
        })
    }
}
