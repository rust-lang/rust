// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code to extract the universally quantified regions declared on a
//! function and the relationships between them. For example:
//!
//! ```
//! fn foo<'a, 'b, 'c: 'b>() { }
//! ```
//!
//! here we would be returning a map assigning each of `{'a, 'b, 'c}`
//! to an index, as well as the `FreeRegionMap` which can compute
//! relationships between them.
//!
//! The code in this file doesn't *do anything* with those results; it
//! just returns them for other code to use.

use rustc::hir::{BodyOwnerKind, HirId};
use rustc::hir::def_id::DefId;
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};
use rustc::infer::region_constraints::GenericKind;
use rustc::infer::outlives::bounds::{self, OutlivesBound};
use rustc::infer::outlives::free_region_map::FreeRegionRelations;
use rustc::ty::{self, RegionVid, Ty, TyCtxt};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::Substs;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::transitive_relation::TransitiveRelation;
use std::iter;
use syntax::ast;

use super::ToRegionVid;

#[derive(Debug)]
pub struct UniversalRegions<'tcx> {
    indices: UniversalRegionIndices<'tcx>,

    /// The vid assigned to `'static`
    pub fr_static: RegionVid,

    /// A special region vid created to represent the current MIR fn
    /// body.  It will outlive the entire CFG but it will not outlive
    /// any other universal regions.
    pub fr_fn_body: RegionVid,

    /// We create region variables such that they are ordered by their
    /// `RegionClassification`. The first block are globals, then
    /// externals, then locals. So things from:
    /// - `FIRST_GLOBAL_INDEX..first_extern_index` are global;
    /// - `first_extern_index..first_local_index` are external; and
    /// - first_local_index..num_universals` are local.
    first_extern_index: usize,

    /// See `first_extern_index`.
    first_local_index: usize,

    /// The total number of universal region variables instantiated.
    num_universals: usize,

    /// The "defining" type for this function, with all universal
    /// regions instantiated.  For a closure or generator, this is the
    /// closure type, but for a top-level function it's the `TyFnDef`.
    pub defining_ty: DefiningTy<'tcx>,

    /// The return type of this function, with all regions replaced by
    /// their universal `RegionVid` equivalents.
    ///
    /// NB. Associated types in this type have not been normalized,
    /// as the name suggests. =)
    pub unnormalized_output_ty: Ty<'tcx>,

    /// The fully liberated input types of this function, with all
    /// regions replaced by their universal `RegionVid` equivalents.
    ///
    /// NB. Associated types in these types have not been normalized,
    /// as the name suggests. =)
    pub unnormalized_input_tys: &'tcx [Ty<'tcx>],

    /// Each RBP `('a, GK)` indicates that `GK: 'a` can be assumed to
    /// be true. These encode relationships like `T: 'a` that are
    /// added via implicit bounds.
    ///
    /// Each region here is guaranteed to be a key in the `indices`
    /// map.  We use the "original" regions (i.e., the keys from the
    /// map, and not the values) because the code in
    /// `process_registered_region_obligations` has some special-cased
    /// logic expecting to see (e.g.) `ReStatic`, and if we supplied
    /// our special inference variable there, we would mess that up.
    pub region_bound_pairs: Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>,

    relations: UniversalRegionRelations,
}

/// The "defining type" for this MIR. The key feature of the "defining
/// type" is that it contains the information needed to derive all the
/// universal regions that are in scope as well as the types of the
/// inputs/output from the MIR. In general, early-bound universal
/// regions appear free in the defining type and late-bound regions
/// appear bound in the signature.
#[derive(Copy, Clone, Debug)]
pub enum DefiningTy<'tcx> {
    /// The MIR is a closure. The signature is found via
    /// `ClosureSubsts::closure_sig_ty`.
    Closure(DefId, ty::ClosureSubsts<'tcx>),

    /// The MIR is a generator. The signature is that generators take
    /// no parameters and return the result of
    /// `ClosureSubsts::generator_return_ty`.
    Generator(DefId, ty::ClosureSubsts<'tcx>, ty::GeneratorInterior<'tcx>),

    /// The MIR is a fn item with the given def-id and substs. The signature
    /// of the function can be bound then with the `fn_sig` query.
    FnDef(DefId, &'tcx Substs<'tcx>),

    /// The MIR represents some form of constant. The signature then
    /// is that it has no inputs and a single return value, which is
    /// the value of the constant.
    Const(Ty<'tcx>),
}

#[derive(Debug)]
struct UniversalRegionIndices<'tcx> {
    /// For those regions that may appear in the parameter environment
    /// ('static and early-bound regions), we maintain a map from the
    /// `ty::Region` to the internal `RegionVid` we are using. This is
    /// used because trait matching and type-checking will feed us
    /// region constraints that reference those regions and we need to
    /// be able to map them our internal `RegionVid`. This is
    /// basically equivalent to a `Substs`, except that it also
    /// contains an entry for `ReStatic` -- it might be nice to just
    /// use a substs, and then handle `ReStatic` another way.
    indices: FxHashMap<ty::Region<'tcx>, RegionVid>,
}

#[derive(Debug)]
struct UniversalRegionRelations {
    /// Stores the outlives relations that are known to hold from the
    /// implied bounds, in-scope where clauses, and that sort of
    /// thing.
    outlives: TransitiveRelation<RegionVid>,

    /// This is the `<=` relation; that is, if `a: b`, then `b <= a`,
    /// and we store that here. This is useful when figuring out how
    /// to express some local region in terms of external regions our
    /// caller will understand.
    inverse_outlives: TransitiveRelation<RegionVid>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionClassification {
    /// A **global** region is one that can be named from
    /// anywhere. There is only one, `'static`.
    Global,

    /// An **external** region is only relevant for closures. In that
    /// case, it refers to regions that are free in the closure type
    /// -- basically, something bound in the surrounding context.
    ///
    /// Consider this example:
    ///
    /// ```
    /// fn foo<'a, 'b>(a: &'a u32, b: &'b u32, c: &'static u32) {
    ///   let closure = for<'x> |x: &'x u32| { .. };
    ///                 ^^^^^^^ pretend this were legal syntax
    ///                         for declaring a late-bound region in
    ///                         a closure signature
    /// }
    /// ```
    ///
    /// Here, the lifetimes `'a` and `'b` would be **external** to the
    /// closure.
    ///
    /// If we are not analyzing a closure, there are no external
    /// lifetimes.
    External,

    /// A **local** lifetime is one about which we know the full set
    /// of relevant constraints (that is, relationships to other named
    /// regions).  For a closure, this includes any region bound in
    /// the closure's signature.  For a fn item, this includes all
    /// regions other than global ones.
    ///
    /// Continuing with the example from `External`, if we were
    /// analyzing the closure, then `'x` would be local (and `'a` and
    /// `'b` are external).  If we are analyzing the function item
    /// `foo`, then `'a` and `'b` are local (and `'x` is not in
    /// scope).
    Local,
}

const FIRST_GLOBAL_INDEX: usize = 0;

impl<'tcx> UniversalRegions<'tcx> {
    /// Creates a new and fully initialized `UniversalRegions` that
    /// contains indices for all the free regions found in the given
    /// MIR -- that is, all the regions that appear in the function's
    /// signature. This will also compute the relationships that are
    /// known between those regions.
    pub fn new(
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let tcx = infcx.tcx;
        let mir_node_id = tcx.hir.as_local_node_id(mir_def_id).unwrap();
        let mir_hir_id = tcx.hir.node_to_hir_id(mir_node_id);
        UniversalRegionsBuilder {
            infcx,
            mir_def_id,
            mir_node_id,
            mir_hir_id,
            param_env,
            region_bound_pairs: vec![],
            relations: UniversalRegionRelations {
                outlives: TransitiveRelation::new(),
                inverse_outlives: TransitiveRelation::new(),
            },
        }.build()
    }

    /// Given a reference to a closure type, extracts all the values
    /// from its free regions and returns a vector with them. This is
    /// used when the closure's creator checks that the
    /// `ClosureRegionRequirements` are met. The requirements from
    /// `ClosureRegionRequirements` are expressed in terms of
    /// `RegionVid` entries that map into the returned vector `V`: so
    /// if the `ClosureRegionRequirements` contains something like
    /// `'1: '2`, then the caller would impose the constraint that
    /// `V[1]: V[2]`.
    pub fn closure_mapping(
        infcx: &InferCtxt<'_, '_, 'tcx>,
        closure_ty: Ty<'tcx>,
        expected_num_vars: usize,
    ) -> IndexVec<RegionVid, ty::Region<'tcx>> {
        let mut region_mapping = IndexVec::with_capacity(expected_num_vars);
        region_mapping.push(infcx.tcx.types.re_static);
        infcx.tcx.for_each_free_region(&closure_ty, |fr| {
            region_mapping.push(fr);
        });

        assert_eq!(
            region_mapping.len(),
            expected_num_vars,
            "index vec had unexpected number of variables"
        );

        region_mapping
    }

    /// True if `r` is a member of this set of universal regions.
    pub fn is_universal_region(&self, r: RegionVid) -> bool {
        (FIRST_GLOBAL_INDEX..self.num_universals).contains(r.index())
    }

    /// Classifies `r` as a universal region, returning `None` if this
    /// is not a member of this set of universal regions.
    pub fn region_classification(&self, r: RegionVid) -> Option<RegionClassification> {
        let index = r.index();
        if (FIRST_GLOBAL_INDEX..self.first_extern_index).contains(index) {
            Some(RegionClassification::Global)
        } else if (self.first_extern_index..self.first_local_index).contains(index) {
            Some(RegionClassification::External)
        } else if (self.first_local_index..self.num_universals).contains(index) {
            Some(RegionClassification::Local)
        } else {
            None
        }
    }

    /// Returns an iterator over all the RegionVids corresponding to
    /// universally quantified free regions.
    pub fn universal_regions(&self) -> impl Iterator<Item = RegionVid> {
        (FIRST_GLOBAL_INDEX..self.num_universals).map(RegionVid::new)
    }

    /// True if `r` is classified as an local region.
    pub fn is_local_free_region(&self, r: RegionVid) -> bool {
        self.region_classification(r) == Some(RegionClassification::Local)
    }

    /// Returns the number of universal regions created in any category.
    pub fn len(&self) -> usize {
        self.num_universals
    }

    /// Given two universal regions, returns the postdominating
    /// upper-bound (effectively the least upper bound).
    ///
    /// (See `TransitiveRelation::postdom_upper_bound` for details on
    /// the postdominating upper bound in general.)
    pub fn postdom_upper_bound(&self, fr1: RegionVid, fr2: RegionVid) -> RegionVid {
        assert!(self.is_universal_region(fr1));
        assert!(self.is_universal_region(fr2));
        *self.relations
            .inverse_outlives
            .postdom_upper_bound(&fr1, &fr2)
            .unwrap_or(&self.fr_static)
    }

    /// Finds an "upper bound" for `fr` that is not local. In other
    /// words, returns the smallest (*) known region `fr1` that (a)
    /// outlives `fr` and (b) is not local. This cannot fail, because
    /// we will always find `'static` at worst.
    ///
    /// (*) If there are multiple competing choices, we pick the "postdominating"
    /// one. See `TransitiveRelation::postdom_upper_bound` for details.
    pub fn non_local_upper_bound(&self, fr: RegionVid) -> RegionVid {
        debug!("non_local_upper_bound(fr={:?})", fr);
        self.non_local_bound(&self.relations.inverse_outlives, fr)
            .unwrap_or(self.fr_static)
    }

    /// Finds a "lower bound" for `fr` that is not local. In other
    /// words, returns the largest (*) known region `fr1` that (a) is
    /// outlived by `fr` and (b) is not local. This cannot fail,
    /// because we will always find `'static` at worst.
    ///
    /// (*) If there are multiple competing choices, we pick the "postdominating"
    /// one. See `TransitiveRelation::postdom_upper_bound` for details.
    pub fn non_local_lower_bound(&self, fr: RegionVid) -> Option<RegionVid> {
        debug!("non_local_lower_bound(fr={:?})", fr);
        self.non_local_bound(&self.relations.outlives, fr)
    }

    /// Returns the number of global plus external universal regions.
    /// For closures, these are the regions that appear free in the
    /// closure type (versus those bound in the closure
    /// signature). They are therefore the regions between which the
    /// closure may impose constraints that its creator must verify.
    pub fn num_global_and_external_regions(&self) -> usize {
        self.first_local_index
    }

    /// Helper for `non_local_upper_bound` and
    /// `non_local_lower_bound`.  Repeatedly invokes `postdom_parent`
    /// until we find something that is not local. Returns None if we
    /// never do so.
    fn non_local_bound(
        &self,
        relation: &TransitiveRelation<RegionVid>,
        fr0: RegionVid,
    ) -> Option<RegionVid> {
        // This method assumes that `fr0` is one of the universally
        // quantified region variables.
        assert!(self.is_universal_region(fr0));

        let mut external_parents = vec![];
        let mut queue = vec![&fr0];

        // Keep expanding `fr` into its parents until we reach
        // non-local regions.
        while let Some(fr) = queue.pop() {
            if !self.is_local_free_region(*fr) {
                external_parents.push(fr);
                continue;
            }

            queue.extend(relation.parents(fr));
        }

        debug!("non_local_bound: external_parents={:?}", external_parents);

        // In case we find more than one, reduce to one for
        // convenience.  This is to prevent us from generating more
        // complex constraints, but it will cause spurious errors.
        let post_dom = relation
            .mutual_immediate_postdominator(external_parents)
            .cloned();

        debug!("non_local_bound: post_dom={:?}", post_dom);

        post_dom.and_then(|post_dom| {
            // If the mutual immediate postdom is not local, then
            // there is no non-local result we can return.
            if !self.is_local_free_region(post_dom) {
                Some(post_dom)
            } else {
                None
            }
        })
    }

    /// True if fr1 is known to outlive fr2.
    ///
    /// This will only ever be true for universally quantified regions.
    pub fn outlives(&self, fr1: RegionVid, fr2: RegionVid) -> bool {
        self.relations.outlives.contains(&fr1, &fr2)
    }

    /// Returns a vector of free regions `x` such that `fr1: x` is
    /// known to hold.
    pub fn regions_outlived_by(&self, fr1: RegionVid) -> Vec<&RegionVid> {
        self.relations.outlives.reachable_from(&fr1)
    }

    /// Get an iterator over all the early-bound regions that have names.
    pub fn named_universal_regions<'s>(
        &'s self,
    ) -> impl Iterator<Item = (ty::Region<'tcx>, ty::RegionVid)> + 's {
        self.indices.indices.iter().map(|(&r, &v)| (r, v))
    }

    /// See `UniversalRegionIndices::to_region_vid`.
    pub fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.indices.to_region_vid(r)
    }
}

struct UniversalRegionsBuilder<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    mir_def_id: DefId,
    mir_hir_id: HirId,
    mir_node_id: ast::NodeId,
    param_env: ty::ParamEnv<'tcx>,
    region_bound_pairs: Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>,
    relations: UniversalRegionRelations,
}

const FR: NLLRegionVariableOrigin = NLLRegionVariableOrigin::FreeRegion;

impl<'cx, 'gcx, 'tcx> UniversalRegionsBuilder<'cx, 'gcx, 'tcx> {
    fn build(mut self) -> UniversalRegions<'tcx> {
        debug!("build(mir_def_id={:?})", self.mir_def_id);

        let param_env = self.param_env;
        debug!("build: param_env={:?}", param_env);

        assert_eq!(FIRST_GLOBAL_INDEX, self.infcx.num_region_vars());

        // Create the "global" region that is always free in all contexts: 'static.
        let fr_static = self.infcx.next_nll_region_var(FR).to_region_vid();

        // We've now added all the global regions. The next ones we
        // add will be external.
        let first_extern_index = self.infcx.num_region_vars();

        let defining_ty = self.defining_ty();
        debug!("build: defining_ty={:?}", defining_ty);

        let mut indices = self.compute_indices(fr_static, defining_ty);
        debug!("build: indices={:?}", indices);

        let bound_inputs_and_output = self.compute_inputs_and_output(&indices, defining_ty);

        // "Liberate" the late-bound regions. These correspond to
        // "local" free regions.
        let first_local_index = self.infcx.num_region_vars();
        let inputs_and_output = self.infcx.replace_bound_regions_with_nll_infer_vars(
            FR,
            self.mir_def_id,
            &bound_inputs_and_output,
            &mut indices,
        );
        let fr_fn_body = self.infcx.next_nll_region_var(FR).to_region_vid();
        let num_universals = self.infcx.num_region_vars();

        // Insert the facts we know from the predicates. Why? Why not.
        self.add_outlives_bounds(&indices, bounds::explicit_outlives_bounds(param_env));

        // Add the implied bounds from inputs and outputs.
        for ty in inputs_and_output {
            debug!("build: input_or_output={:?}", ty);
            self.add_implied_bounds(&indices, ty);
        }

        // Finally:
        // - outlives is reflexive, so `'r: 'r` for every region `'r`
        // - `'static: 'r` for every region `'r`
        // - `'r: 'fn_body` for every (other) universally quantified
        //   region `'r`, all of which are provided by our caller
        for fr in (FIRST_GLOBAL_INDEX..num_universals).map(RegionVid::new) {
            debug!(
                "build: relating free region {:?} to itself and to 'static",
                fr
            );
            self.relations.relate_universal_regions(fr, fr);
            self.relations.relate_universal_regions(fr_static, fr);
            self.relations.relate_universal_regions(fr, fr_fn_body);
        }

        let (unnormalized_output_ty, unnormalized_input_tys) =
            inputs_and_output.split_last().unwrap();

        debug!(
            "build: global regions = {}..{}",
            FIRST_GLOBAL_INDEX,
            first_extern_index
        );
        debug!(
            "build: extern regions = {}..{}",
            first_extern_index,
            first_local_index
        );
        debug!(
            "build: local regions  = {}..{}",
            first_local_index,
            num_universals
        );

        UniversalRegions {
            indices,
            fr_static,
            fr_fn_body,
            first_extern_index,
            first_local_index,
            num_universals,
            defining_ty,
            unnormalized_output_ty,
            unnormalized_input_tys,
            region_bound_pairs: self.region_bound_pairs,
            relations: self.relations,
        }
    }

    /// Returns the "defining type" of the current MIR;
    /// see `DefiningTy` for details.
    fn defining_ty(&self) -> DefiningTy<'tcx> {
        let tcx = self.infcx.tcx;

        let closure_base_def_id = tcx.closure_base_def_id(self.mir_def_id);

        let defining_ty = if self.mir_def_id == closure_base_def_id {
            tcx.type_of(closure_base_def_id)
        } else {
            let tables = tcx.typeck_tables_of(self.mir_def_id);
            tables.node_id_to_type(self.mir_hir_id)
        };

        let defining_ty = self.infcx
            .replace_free_regions_with_nll_infer_vars(FR, &defining_ty);

        match tcx.hir.body_owner_kind(self.mir_node_id) {
            BodyOwnerKind::Fn => match defining_ty.sty {
                ty::TyClosure(def_id, substs) => DefiningTy::Closure(def_id, substs),
                ty::TyGenerator(def_id, substs, interior) => {
                    DefiningTy::Generator(def_id, substs, interior)
                }
                ty::TyFnDef(def_id, substs) => DefiningTy::FnDef(def_id, substs),
                _ => span_bug!(
                    tcx.def_span(self.mir_def_id),
                    "expected defining type for `{:?}`: `{:?}`",
                    self.mir_def_id,
                    defining_ty
                ),
            },
            BodyOwnerKind::Const | BodyOwnerKind::Static(..) => DefiningTy::Const(defining_ty),
        }
    }

    /// Builds a hashmap that maps from the universal regions that are
    /// in scope (as a `ty::Region<'tcx>`) to their indices (as a
    /// `RegionVid`). The map returned by this function contains only
    /// the early-bound regions.
    fn compute_indices(
        &self,
        fr_static: RegionVid,
        defining_ty: DefiningTy<'tcx>,
    ) -> UniversalRegionIndices<'tcx> {
        let tcx = self.infcx.tcx;
        let gcx = tcx.global_tcx();
        let closure_base_def_id = tcx.closure_base_def_id(self.mir_def_id);
        let identity_substs = Substs::identity_for_item(gcx, closure_base_def_id);
        let fr_substs = match defining_ty {
            DefiningTy::Closure(_, substs) | DefiningTy::Generator(_, substs, _) => {
                // In the case of closures, we rely on the fact that
                // the first N elements in the ClosureSubsts are
                // inherited from the `closure_base_def_id`.
                // Therefore, when we zip together (below) with
                // `identity_substs`, we will get only those regions
                // that correspond to early-bound regions declared on
                // the `closure_base_def_id`.
                assert!(substs.substs.len() >= identity_substs.len());
                assert_eq!(substs.substs.regions().count(), identity_substs.regions().count());
                substs.substs
            }

            DefiningTy::FnDef(_, substs) => substs,

            // When we encounter other sorts of constant
            // expressions, such as the `22` in `[foo; 22]`, we can
            // get the type `usize` here. For now, just return an
            // empty vector of substs in this case, since there are no
            // generics in scope in such expressions right now.
            DefiningTy::Const(_) => {
                assert!(identity_substs.is_empty());
                identity_substs
            }
        };

        let global_mapping = iter::once((gcx.types.re_static, fr_static));
        let subst_mapping = identity_substs
            .regions()
            .zip(fr_substs.regions().map(|r| r.to_region_vid()));

        UniversalRegionIndices {
            indices: global_mapping.chain(subst_mapping).collect(),
        }
    }

    fn compute_inputs_and_output(
        &self,
        indices: &UniversalRegionIndices<'tcx>,
        defining_ty: DefiningTy<'tcx>,
    ) -> ty::Binder<&'tcx ty::Slice<Ty<'tcx>>> {
        let tcx = self.infcx.tcx;
        match defining_ty {
            DefiningTy::Closure(def_id, substs) => {
                assert_eq!(self.mir_def_id, def_id);
                let closure_sig = substs.closure_sig_ty(def_id, tcx).fn_sig(tcx);
                let inputs_and_output = closure_sig.inputs_and_output();
                let closure_ty = tcx.closure_env_ty(def_id, substs).unwrap();
                ty::Binder::fuse(
                    closure_ty,
                    inputs_and_output,
                    |closure_ty, inputs_and_output| {
                        // The "inputs" of the closure in the
                        // signature appear as a tuple.  The MIR side
                        // flattens this tuple.
                        let (&output, tuplized_inputs) = inputs_and_output.split_last().unwrap();
                        assert_eq!(tuplized_inputs.len(), 1, "multiple closure inputs");
                        let inputs = match tuplized_inputs[0].sty {
                            ty::TyTuple(inputs, _) => inputs,
                            _ => bug!("closure inputs not a tuple: {:?}", tuplized_inputs[0]),
                        };

                        tcx.mk_type_list(
                            iter::once(closure_ty)
                                .chain(inputs.iter().cloned())
                                .chain(iter::once(output)),
                        )
                    },
                )
            }

            DefiningTy::Generator(def_id, substs, interior) => {
                assert_eq!(self.mir_def_id, def_id);
                let output = substs.generator_return_ty(def_id, tcx);
                let generator_ty = tcx.mk_generator(def_id, substs, interior);
                let inputs_and_output = self.infcx.tcx.intern_type_list(&[generator_ty, output]);
                ty::Binder::dummy(inputs_and_output)
            }

            DefiningTy::FnDef(def_id, _) => {
                let sig = tcx.fn_sig(def_id);
                let sig = indices.fold_to_region_vids(tcx, &sig);
                sig.inputs_and_output()
            }

            // This happens on things like `[foo; 22]`. Hence, no
            // inputs, one output, but it seems like we need a more
            // general way to handle this category of MIR.
            DefiningTy::Const(ty) => ty::Binder::dummy(tcx.mk_type_list(iter::once(ty))),
        }
    }

    /// Update the type of a single local, which should represent
    /// either the return type of the MIR or one of its arguments. At
    /// the same time, compute and add any implied bounds that come
    /// from this local.
    ///
    /// Assumes that `universal_regions` indices map is fully constructed.
    fn add_implied_bounds(&mut self, indices: &UniversalRegionIndices<'tcx>, ty: Ty<'tcx>) {
        debug!("add_implied_bounds(ty={:?})", ty);
        let span = self.infcx.tcx.def_span(self.mir_def_id);
        let bounds = self.infcx
            .implied_outlives_bounds(self.param_env, self.mir_node_id, ty, span);
        self.add_outlives_bounds(indices, bounds);
    }

    /// Registers the `OutlivesBound` items from `outlives_bounds` in
    /// the outlives relation as well as the region-bound pairs
    /// listing.
    fn add_outlives_bounds<I>(&mut self, indices: &UniversalRegionIndices<'tcx>, outlives_bounds: I)
    where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds(bound={:?})", outlives_bound);

            match outlives_bound {
                OutlivesBound::RegionSubRegion(r1, r2) => {
                    // The bound says that `r1 <= r2`; we store `r2: r1`.
                    let r1 = indices.to_region_vid(r1);
                    let r2 = indices.to_region_vid(r2);
                    self.relations.relate_universal_regions(r2, r1);
                }

                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.region_bound_pairs
                        .push((r_a, GenericKind::Param(param_b)));
                }

                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.region_bound_pairs
                        .push((r_a, GenericKind::Projection(projection_b)));
                }
            }
        }
    }
}

impl UniversalRegionRelations {
    /// Records in the `outlives_relation` (and
    /// `inverse_outlives_relation`) that `fr_a: fr_b`.
    fn relate_universal_regions(&mut self, fr_a: RegionVid, fr_b: RegionVid) {
        debug!(
            "relate_universal_regions: fr_a={:?} outlives fr_b={:?}",
            fr_a,
            fr_b
        );
        self.outlives.add(fr_a, fr_b);
        self.inverse_outlives.add(fr_b, fr_a);
    }
}

trait InferCtxtExt<'tcx> {
    fn replace_free_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        value: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>;

    fn replace_bound_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        all_outlive_scope: DefId,
        value: &ty::Binder<T>,
        indices: &mut UniversalRegionIndices<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>;
}

impl<'cx, 'gcx, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'cx, 'gcx, 'tcx> {
    fn replace_free_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        value: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.tcx.fold_regions(value, &mut false, |_region, _depth| {
            self.next_nll_region_var(origin)
        })
    }

    fn replace_bound_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        all_outlive_scope: DefId,
        value: &ty::Binder<T>,
        indices: &mut UniversalRegionIndices<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let (value, _map) = self.tcx.replace_late_bound_regions(value, |br| {
            let liberated_region = self.tcx.mk_region(ty::ReFree(ty::FreeRegion {
                scope: all_outlive_scope,
                bound_region: br,
            }));
            let region_vid = self.next_nll_region_var(origin);
            indices.insert_late_bound_region(liberated_region, region_vid.to_region_vid());
            region_vid
        });
        value
    }
}

impl<'tcx> UniversalRegionIndices<'tcx> {
    /// Initially, the `UniversalRegionIndices` map contains only the
    /// early-bound regions in scope. Once that is all setup, we come
    /// in later and instantiate the late-bound regions, and then we
    /// insert the `ReFree` version of those into the map as
    /// well. These are used for error reporting.
    fn insert_late_bound_region(&mut self, r: ty::Region<'tcx>,
                                vid: ty::RegionVid)
    {
        self.indices.insert(r, vid);
    }

    /// Converts `r` into a local inference variable: `r` can either
    /// by a `ReVar` (i.e., already a reference to an inference
    /// variable) or it can be `'static` or some early-bound
    /// region. This is useful when taking the results from
    /// type-checking and trait-matching, which may sometimes
    /// reference those regions from the `ParamEnv`. It is also used
    /// during initialization. Relies on the `indices` map having been
    /// fully initialized.
    pub fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        match r {
            ty::ReEarlyBound(..) | ty::ReStatic => *self.indices.get(&r).unwrap(),
            ty::ReVar(..) => r.to_region_vid(),
            _ => bug!("cannot convert `{:?}` to a region vid", r),
        }
    }

    /// Replace all free regions in `value` with region vids, as
    /// returned by `to_region_vid`.
    pub fn fold_to_region_vids<T>(&self, tcx: TyCtxt<'_, '_, 'tcx>, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        tcx.fold_regions(value, &mut false, |region, _| {
            tcx.mk_region(ty::ReVar(self.to_region_vid(region)))
        })
    }
}

/// This trait is used by the `impl-trait` constraint code to abstract
/// over the `FreeRegionMap` from lexical regions and
/// `UniversalRegions` (from NLL)`.
impl<'tcx> FreeRegionRelations<'tcx> for UniversalRegions<'tcx> {
    fn sub_free_regions(&self, shorter: ty::Region<'tcx>, longer: ty::Region<'tcx>) -> bool {
        let shorter = shorter.to_region_vid();
        assert!(self.is_universal_region(shorter));
        let longer = longer.to_region_vid();
        assert!(self.is_universal_region(longer));
        self.outlives(longer, shorter)
    }
}
