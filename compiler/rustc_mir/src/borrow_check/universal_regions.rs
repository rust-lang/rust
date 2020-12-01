//! Code to extract the universally quantified regions declared on a
//! function and the relationships between them. For example:
//!
//! ```
//! fn foo<'a, 'b, 'c: 'b>() { }
//! ```
//!
//! here we would return a map assigning each of `{'a, 'b, 'c}`
//! to an index, as well as the `FreeRegionMap` which can compute
//! relationships between them.
//!
//! The code in this file doesn't *do anything* with those results; it
//! just returns them for other code to use.

use either::Either;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{BodyOwnerKind, HirId};
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::{InferCtxt, NLLRegionVariableOrigin};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt};
use std::iter;

use crate::borrow_check::nll::ToRegionVid;

#[derive(Debug)]
pub struct UniversalRegions<'tcx> {
    indices: UniversalRegionIndices<'tcx>,

    /// The vid assigned to `'static`
    pub fr_static: RegionVid,

    /// A special region vid created to represent the current MIR fn
    /// body. It will outlive the entire CFG but it will not outlive
    /// any other universal regions.
    pub fr_fn_body: RegionVid,

    /// We create region variables such that they are ordered by their
    /// `RegionClassification`. The first block are globals, then
    /// externals, then locals. So, things from:
    /// - `FIRST_GLOBAL_INDEX..first_extern_index` are global,
    /// - `first_extern_index..first_local_index` are external,
    /// - `first_local_index..num_universals` are local.
    first_extern_index: usize,

    /// See `first_extern_index`.
    first_local_index: usize,

    /// The total number of universal region variables instantiated.
    num_universals: usize,

    /// A special region variable created for the `'empty(U0)` region.
    /// Note that this is **not** a "universal" region, as it doesn't
    /// represent a universally bound placeholder or any such thing.
    /// But we do create it here in this type because it's a useful region
    /// to have around in a few limited cases.
    pub root_empty: RegionVid,

    /// The "defining" type for this function, with all universal
    /// regions instantiated. For a closure or generator, this is the
    /// closure type, but for a top-level function it's the `FnDef`.
    pub defining_ty: DefiningTy<'tcx>,

    /// The return type of this function, with all regions replaced by
    /// their universal `RegionVid` equivalents.
    ///
    /// N.B., associated types in this type have not been normalized,
    /// as the name suggests. =)
    pub unnormalized_output_ty: Ty<'tcx>,

    /// The fully liberated input types of this function, with all
    /// regions replaced by their universal `RegionVid` equivalents.
    ///
    /// N.B., associated types in these types have not been normalized,
    /// as the name suggests. =)
    pub unnormalized_input_tys: &'tcx [Ty<'tcx>],

    pub yield_ty: Option<Ty<'tcx>>,
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
    Closure(DefId, SubstsRef<'tcx>),

    /// The MIR is a generator. The signature is that generators take
    /// no parameters and return the result of
    /// `ClosureSubsts::generator_return_ty`.
    Generator(DefId, SubstsRef<'tcx>, hir::Movability),

    /// The MIR is a fn item with the given `DefId` and substs. The signature
    /// of the function can be bound then with the `fn_sig` query.
    FnDef(DefId, SubstsRef<'tcx>),

    /// The MIR represents some form of constant. The signature then
    /// is that it has no inputs and a single return value, which is
    /// the value of the constant.
    Const(DefId, SubstsRef<'tcx>),
}

impl<'tcx> DefiningTy<'tcx> {
    /// Returns a list of all the upvar types for this MIR. If this is
    /// not a closure or generator, there are no upvars, and hence it
    /// will be an empty list. The order of types in this list will
    /// match up with the upvar order in the HIR, typesystem, and MIR.
    pub fn upvar_tys(self) -> impl Iterator<Item = Ty<'tcx>> + 'tcx {
        match self {
            DefiningTy::Closure(_, substs) => Either::Left(substs.as_closure().upvar_tys()),
            DefiningTy::Generator(_, substs, _) => {
                Either::Right(Either::Left(substs.as_generator().upvar_tys()))
            }
            DefiningTy::FnDef(..) | DefiningTy::Const(..) => {
                Either::Right(Either::Right(iter::empty()))
            }
        }
    }

    /// Number of implicit inputs -- notably the "environment"
    /// parameter for closures -- that appear in MIR but not in the
    /// user's code.
    pub fn implicit_inputs(self) -> usize {
        match self {
            DefiningTy::Closure(..) | DefiningTy::Generator(..) => 1,
            DefiningTy::FnDef(..) | DefiningTy::Const(..) => 0,
        }
    }

    pub fn is_fn_def(&self) -> bool {
        match *self {
            DefiningTy::FnDef(..) => true,
            _ => false,
        }
    }

    pub fn is_const(&self) -> bool {
        match *self {
            DefiningTy::Const(..) => true,
            _ => false,
        }
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            DefiningTy::Closure(def_id, ..)
            | DefiningTy::Generator(def_id, ..)
            | DefiningTy::FnDef(def_id, ..)
            | DefiningTy::Const(def_id, ..) => def_id,
        }
    }
}

#[derive(Debug)]
struct UniversalRegionIndices<'tcx> {
    /// For those regions that may appear in the parameter environment
    /// ('static and early-bound regions), we maintain a map from the
    /// `ty::Region` to the internal `RegionVid` we are using. This is
    /// used because trait matching and type-checking will feed us
    /// region constraints that reference those regions and we need to
    /// be able to map them our internal `RegionVid`. This is
    /// basically equivalent to a `InternalSubsts`, except that it also
    /// contains an entry for `ReStatic` -- it might be nice to just
    /// use a substs, and then handle `ReStatic` another way.
    indices: FxHashMap<ty::Region<'tcx>, RegionVid>,
}

#[derive(Debug, PartialEq)]
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
    /// regions). For a closure, this includes any region bound in
    /// the closure's signature. For a fn item, this includes all
    /// regions other than global ones.
    ///
    /// Continuing with the example from `External`, if we were
    /// analyzing the closure, then `'x` would be local (and `'a` and
    /// `'b` are external). If we are analyzing the function item
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
        infcx: &InferCtxt<'_, 'tcx>,
        mir_def: ty::WithOptConstParam<LocalDefId>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let tcx = infcx.tcx;
        let mir_hir_id = tcx.hir().local_def_id_to_hir_id(mir_def.did);
        UniversalRegionsBuilder { infcx, mir_def, mir_hir_id, param_env }.build()
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
        tcx: TyCtxt<'tcx>,
        closure_substs: SubstsRef<'tcx>,
        expected_num_vars: usize,
        closure_base_def_id: DefId,
    ) -> IndexVec<RegionVid, ty::Region<'tcx>> {
        let mut region_mapping = IndexVec::with_capacity(expected_num_vars);
        region_mapping.push(tcx.lifetimes.re_static);
        tcx.for_each_free_region(&closure_substs, |fr| {
            region_mapping.push(fr);
        });

        for_each_late_bound_region_defined_on(tcx, closure_base_def_id, |r| {
            region_mapping.push(r);
        });

        assert_eq!(
            region_mapping.len(),
            expected_num_vars,
            "index vec had unexpected number of variables"
        );

        region_mapping
    }

    /// Returns `true` if `r` is a member of this set of universal regions.
    pub fn is_universal_region(&self, r: RegionVid) -> bool {
        (FIRST_GLOBAL_INDEX..self.num_universals).contains(&r.index())
    }

    /// Classifies `r` as a universal region, returning `None` if this
    /// is not a member of this set of universal regions.
    pub fn region_classification(&self, r: RegionVid) -> Option<RegionClassification> {
        let index = r.index();
        if (FIRST_GLOBAL_INDEX..self.first_extern_index).contains(&index) {
            Some(RegionClassification::Global)
        } else if (self.first_extern_index..self.first_local_index).contains(&index) {
            Some(RegionClassification::External)
        } else if (self.first_local_index..self.num_universals).contains(&index) {
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

    /// Returns `true` if `r` is classified as an local region.
    pub fn is_local_free_region(&self, r: RegionVid) -> bool {
        self.region_classification(r) == Some(RegionClassification::Local)
    }

    /// Returns the number of universal regions created in any category.
    pub fn len(&self) -> usize {
        self.num_universals
    }

    /// Returns the number of global plus external universal regions.
    /// For closures, these are the regions that appear free in the
    /// closure type (versus those bound in the closure
    /// signature). They are therefore the regions between which the
    /// closure may impose constraints that its creator must verify.
    pub fn num_global_and_external_regions(&self) -> usize {
        self.first_local_index
    }

    /// Gets an iterator over all the early-bound regions that have names.
    pub fn named_universal_regions<'s>(
        &'s self,
    ) -> impl Iterator<Item = (ty::Region<'tcx>, ty::RegionVid)> + 's {
        self.indices.indices.iter().map(|(&r, &v)| (r, v))
    }

    /// See `UniversalRegionIndices::to_region_vid`.
    pub fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        if let ty::ReEmpty(ty::UniverseIndex::ROOT) = r {
            self.root_empty
        } else {
            self.indices.to_region_vid(r)
        }
    }

    /// As part of the NLL unit tests, you can annotate a function with
    /// `#[rustc_regions]`, and we will emit information about the region
    /// inference context and -- in particular -- the external constraints
    /// that this region imposes on others. The methods in this file
    /// handle the part about dumping the inference context internal
    /// state.
    crate fn annotate(&self, tcx: TyCtxt<'tcx>, err: &mut DiagnosticBuilder<'_>) {
        match self.defining_ty {
            DefiningTy::Closure(def_id, substs) => {
                err.note(&format!(
                    "defining type: {} with closure substs {:#?}",
                    tcx.def_path_str_with_substs(def_id, substs),
                    &substs[tcx.generics_of(def_id).parent_count..],
                ));

                // FIXME: It'd be nice to print the late-bound regions
                // here, but unfortunately these wind up stored into
                // tests, and the resulting print-outs include def-ids
                // and other things that are not stable across tests!
                // So we just include the region-vid. Annoying.
                let closure_base_def_id = tcx.closure_base_def_id(def_id);
                for_each_late_bound_region_defined_on(tcx, closure_base_def_id, |r| {
                    err.note(&format!("late-bound region is {:?}", self.to_region_vid(r),));
                });
            }
            DefiningTy::Generator(def_id, substs, _) => {
                err.note(&format!(
                    "defining type: {} with generator substs {:#?}",
                    tcx.def_path_str_with_substs(def_id, substs),
                    &substs[tcx.generics_of(def_id).parent_count..],
                ));

                // FIXME: As above, we'd like to print out the region
                // `r` but doing so is not stable across architectures
                // and so forth.
                let closure_base_def_id = tcx.closure_base_def_id(def_id);
                for_each_late_bound_region_defined_on(tcx, closure_base_def_id, |r| {
                    err.note(&format!("late-bound region is {:?}", self.to_region_vid(r),));
                });
            }
            DefiningTy::FnDef(def_id, substs) => {
                err.note(&format!(
                    "defining type: {}",
                    tcx.def_path_str_with_substs(def_id, substs),
                ));
            }
            DefiningTy::Const(def_id, substs) => {
                err.note(&format!(
                    "defining constant type: {}",
                    tcx.def_path_str_with_substs(def_id, substs),
                ));
            }
        }
    }
}

struct UniversalRegionsBuilder<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    mir_def: ty::WithOptConstParam<LocalDefId>,
    mir_hir_id: HirId,
    param_env: ty::ParamEnv<'tcx>,
}

const FR: NLLRegionVariableOrigin = NLLRegionVariableOrigin::FreeRegion;

impl<'cx, 'tcx> UniversalRegionsBuilder<'cx, 'tcx> {
    fn build(self) -> UniversalRegions<'tcx> {
        debug!("build(mir_def={:?})", self.mir_def);

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

        let closure_base_def_id = self.infcx.tcx.closure_base_def_id(self.mir_def.did.to_def_id());

        // If this is a closure or generator, then the late-bound regions from the enclosing
        // function are actually external regions to us. For example, here, 'a is not local
        // to the closure c (although it is local to the fn foo):
        // fn foo<'a>() {
        //     let c = || { let x: &'a u32 = ...; }
        // }
        if self.mir_def.did.to_def_id() != closure_base_def_id {
            self.infcx
                .replace_late_bound_regions_with_nll_infer_vars(self.mir_def.did, &mut indices)
        }

        let bound_inputs_and_output = self.compute_inputs_and_output(&indices, defining_ty);

        // "Liberate" the late-bound regions. These correspond to
        // "local" free regions.
        let first_local_index = self.infcx.num_region_vars();
        let inputs_and_output = self.infcx.replace_bound_regions_with_nll_infer_vars(
            FR,
            self.mir_def.did,
            bound_inputs_and_output,
            &mut indices,
        );
        // Converse of above, if this is a function then the late-bound regions declared on its
        // signature are local to the fn.
        if self.mir_def.did.to_def_id() == closure_base_def_id {
            self.infcx
                .replace_late_bound_regions_with_nll_infer_vars(self.mir_def.did, &mut indices);
        }

        let (unnormalized_output_ty, mut unnormalized_input_tys) =
            inputs_and_output.split_last().unwrap();

        // C-variadic fns also have a `VaList` input that's not listed in the signature
        // (as it's created inside the body itself, not passed in from outside).
        if let DefiningTy::FnDef(def_id, _) = defining_ty {
            if self.infcx.tcx.fn_sig(def_id).c_variadic() {
                let va_list_did = self.infcx.tcx.require_lang_item(
                    LangItem::VaList,
                    Some(self.infcx.tcx.def_span(self.mir_def.did)),
                );
                let region = self
                    .infcx
                    .tcx
                    .mk_region(ty::ReVar(self.infcx.next_nll_region_var(FR).to_region_vid()));
                let va_list_ty =
                    self.infcx.tcx.type_of(va_list_did).subst(self.infcx.tcx, &[region.into()]);

                unnormalized_input_tys = self.infcx.tcx.mk_type_list(
                    unnormalized_input_tys.iter().copied().chain(iter::once(va_list_ty)),
                );
            }
        }

        let fr_fn_body = self.infcx.next_nll_region_var(FR).to_region_vid();
        let num_universals = self.infcx.num_region_vars();

        debug!("build: global regions = {}..{}", FIRST_GLOBAL_INDEX, first_extern_index);
        debug!("build: extern regions = {}..{}", first_extern_index, first_local_index);
        debug!("build: local regions  = {}..{}", first_local_index, num_universals);

        let yield_ty = match defining_ty {
            DefiningTy::Generator(_, substs, _) => Some(substs.as_generator().yield_ty()),
            _ => None,
        };

        let root_empty = self
            .infcx
            .next_nll_region_var(NLLRegionVariableOrigin::RootEmptyRegion)
            .to_region_vid();

        UniversalRegions {
            indices,
            fr_static,
            fr_fn_body,
            root_empty,
            first_extern_index,
            first_local_index,
            num_universals,
            defining_ty,
            unnormalized_output_ty,
            unnormalized_input_tys,
            yield_ty,
        }
    }

    /// Returns the "defining type" of the current MIR;
    /// see `DefiningTy` for details.
    fn defining_ty(&self) -> DefiningTy<'tcx> {
        let tcx = self.infcx.tcx;
        let closure_base_def_id = tcx.closure_base_def_id(self.mir_def.did.to_def_id());

        match tcx.hir().body_owner_kind(self.mir_hir_id) {
            BodyOwnerKind::Closure | BodyOwnerKind::Fn => {
                let defining_ty = if self.mir_def.did.to_def_id() == closure_base_def_id {
                    tcx.type_of(closure_base_def_id)
                } else {
                    let tables = tcx.typeck(self.mir_def.did);
                    tables.node_type(self.mir_hir_id)
                };

                debug!("defining_ty (pre-replacement): {:?}", defining_ty);

                let defining_ty =
                    self.infcx.replace_free_regions_with_nll_infer_vars(FR, defining_ty);

                match *defining_ty.kind() {
                    ty::Closure(def_id, substs) => DefiningTy::Closure(def_id, substs),
                    ty::Generator(def_id, substs, movability) => {
                        DefiningTy::Generator(def_id, substs, movability)
                    }
                    ty::FnDef(def_id, substs) => DefiningTy::FnDef(def_id, substs),
                    _ => span_bug!(
                        tcx.def_span(self.mir_def.did),
                        "expected defining type for `{:?}`: `{:?}`",
                        self.mir_def.did,
                        defining_ty
                    ),
                }
            }

            BodyOwnerKind::Const | BodyOwnerKind::Static(..) => {
                assert_eq!(self.mir_def.did.to_def_id(), closure_base_def_id);
                let identity_substs = InternalSubsts::identity_for_item(tcx, closure_base_def_id);
                let substs =
                    self.infcx.replace_free_regions_with_nll_infer_vars(FR, identity_substs);
                DefiningTy::Const(self.mir_def.did.to_def_id(), substs)
            }
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
        let closure_base_def_id = tcx.closure_base_def_id(self.mir_def.did.to_def_id());
        let identity_substs = InternalSubsts::identity_for_item(tcx, closure_base_def_id);
        let fr_substs = match defining_ty {
            DefiningTy::Closure(_, ref substs) | DefiningTy::Generator(_, ref substs, _) => {
                // In the case of closures, we rely on the fact that
                // the first N elements in the ClosureSubsts are
                // inherited from the `closure_base_def_id`.
                // Therefore, when we zip together (below) with
                // `identity_substs`, we will get only those regions
                // that correspond to early-bound regions declared on
                // the `closure_base_def_id`.
                assert!(substs.len() >= identity_substs.len());
                assert_eq!(substs.regions().count(), identity_substs.regions().count());
                substs
            }

            DefiningTy::FnDef(_, substs) | DefiningTy::Const(_, substs) => substs,
        };

        let global_mapping = iter::once((tcx.lifetimes.re_static, fr_static));
        let subst_mapping =
            identity_substs.regions().zip(fr_substs.regions().map(|r| r.to_region_vid()));

        UniversalRegionIndices { indices: global_mapping.chain(subst_mapping).collect() }
    }

    fn compute_inputs_and_output(
        &self,
        indices: &UniversalRegionIndices<'tcx>,
        defining_ty: DefiningTy<'tcx>,
    ) -> ty::Binder<&'tcx ty::List<Ty<'tcx>>> {
        let tcx = self.infcx.tcx;
        match defining_ty {
            DefiningTy::Closure(def_id, substs) => {
                assert_eq!(self.mir_def.did.to_def_id(), def_id);
                let closure_sig = substs.as_closure().sig();
                let inputs_and_output = closure_sig.inputs_and_output();
                let closure_ty = tcx.closure_env_ty(def_id, substs).unwrap();
                ty::Binder::fuse(closure_ty, inputs_and_output, |closure_ty, inputs_and_output| {
                    // The "inputs" of the closure in the
                    // signature appear as a tuple.  The MIR side
                    // flattens this tuple.
                    let (&output, tuplized_inputs) = inputs_and_output.split_last().unwrap();
                    assert_eq!(tuplized_inputs.len(), 1, "multiple closure inputs");
                    let inputs = match tuplized_inputs[0].kind() {
                        ty::Tuple(inputs) => inputs,
                        _ => bug!("closure inputs not a tuple: {:?}", tuplized_inputs[0]),
                    };

                    tcx.mk_type_list(
                        iter::once(closure_ty)
                            .chain(inputs.iter().map(|k| k.expect_ty()))
                            .chain(iter::once(output)),
                    )
                })
            }

            DefiningTy::Generator(def_id, substs, movability) => {
                assert_eq!(self.mir_def.did.to_def_id(), def_id);
                let resume_ty = substs.as_generator().resume_ty();
                let output = substs.as_generator().return_ty();
                let generator_ty = tcx.mk_generator(def_id, substs, movability);
                let inputs_and_output =
                    self.infcx.tcx.intern_type_list(&[generator_ty, resume_ty, output]);
                ty::Binder::dummy(inputs_and_output)
            }

            DefiningTy::FnDef(def_id, _) => {
                let sig = tcx.fn_sig(def_id);
                let sig = indices.fold_to_region_vids(tcx, sig);
                sig.inputs_and_output()
            }

            DefiningTy::Const(def_id, _) => {
                // For a constant body, there are no inputs, and one
                // "output" (the type of the constant).
                assert_eq!(self.mir_def.did.to_def_id(), def_id);
                let ty = tcx.type_of(self.mir_def.def_id_for_type_of());
                let ty = indices.fold_to_region_vids(tcx, ty);
                ty::Binder::dummy(tcx.intern_type_list(&[ty]))
            }
        }
    }
}

trait InferCtxtExt<'tcx> {
    fn replace_free_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        value: T,
    ) -> T
    where
        T: TypeFoldable<'tcx>;

    fn replace_bound_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        all_outlive_scope: LocalDefId,
        value: ty::Binder<T>,
        indices: &mut UniversalRegionIndices<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>;

    fn replace_late_bound_regions_with_nll_infer_vars(
        &self,
        mir_def_id: LocalDefId,
        indices: &mut UniversalRegionIndices<'tcx>,
    );
}

impl<'cx, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'cx, 'tcx> {
    fn replace_free_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        value: T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.tcx.fold_regions(value, &mut false, |_region, _depth| self.next_nll_region_var(origin))
    }

    fn replace_bound_regions_with_nll_infer_vars<T>(
        &self,
        origin: NLLRegionVariableOrigin,
        all_outlive_scope: LocalDefId,
        value: ty::Binder<T>,
        indices: &mut UniversalRegionIndices<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "replace_bound_regions_with_nll_infer_vars(value={:?}, all_outlive_scope={:?})",
            value, all_outlive_scope,
        );
        let (value, _map) = self.tcx.replace_late_bound_regions(value, |br| {
            debug!("replace_bound_regions_with_nll_infer_vars: br={:?}", br);
            let liberated_region = self.tcx.mk_region(ty::ReFree(ty::FreeRegion {
                scope: all_outlive_scope.to_def_id(),
                bound_region: br,
            }));
            let region_vid = self.next_nll_region_var(origin);
            indices.insert_late_bound_region(liberated_region, region_vid.to_region_vid());
            debug!(
                "replace_bound_regions_with_nll_infer_vars: liberated_region={:?} => {:?}",
                liberated_region, region_vid
            );
            region_vid
        });
        value
    }

    /// Finds late-bound regions that do not appear in the parameter listing and adds them to the
    /// indices vector. Typically, we identify late-bound regions as we process the inputs and
    /// outputs of the closure/function. However, sometimes there are late-bound regions which do
    /// not appear in the fn parameters but which are nonetheless in scope. The simplest case of
    /// this are unused functions, like fn foo<'a>() { } (see e.g., #51351). Despite not being used,
    /// users can still reference these regions (e.g., let x: &'a u32 = &22;), so we need to create
    /// entries for them and store them in the indices map. This code iterates over the complete
    /// set of late-bound regions and checks for any that we have not yet seen, adding them to the
    /// inputs vector.
    fn replace_late_bound_regions_with_nll_infer_vars(
        &self,
        mir_def_id: LocalDefId,
        indices: &mut UniversalRegionIndices<'tcx>,
    ) {
        debug!("replace_late_bound_regions_with_nll_infer_vars(mir_def_id={:?})", mir_def_id);
        let closure_base_def_id = self.tcx.closure_base_def_id(mir_def_id.to_def_id());
        for_each_late_bound_region_defined_on(self.tcx, closure_base_def_id, |r| {
            debug!("replace_late_bound_regions_with_nll_infer_vars: r={:?}", r);
            if !indices.indices.contains_key(&r) {
                let region_vid = self.next_nll_region_var(FR);
                indices.insert_late_bound_region(r, region_vid.to_region_vid());
            }
        });
    }
}

impl<'tcx> UniversalRegionIndices<'tcx> {
    /// Initially, the `UniversalRegionIndices` map contains only the
    /// early-bound regions in scope. Once that is all setup, we come
    /// in later and instantiate the late-bound regions, and then we
    /// insert the `ReFree` version of those into the map as
    /// well. These are used for error reporting.
    fn insert_late_bound_region(&mut self, r: ty::Region<'tcx>, vid: ty::RegionVid) {
        debug!("insert_late_bound_region({:?}, {:?})", r, vid);
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
        if let ty::ReVar(..) = r {
            r.to_region_vid()
        } else {
            *self
                .indices
                .get(&r)
                .unwrap_or_else(|| bug!("cannot convert `{:?}` to a region vid", r))
        }
    }

    /// Replaces all free regions in `value` with region vids, as
    /// returned by `to_region_vid`.
    pub fn fold_to_region_vids<T>(&self, tcx: TyCtxt<'tcx>, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        tcx.fold_regions(value, &mut false, |region, _| {
            tcx.mk_region(ty::ReVar(self.to_region_vid(region)))
        })
    }
}

/// Iterates over the late-bound regions defined on fn_def_id and
/// invokes `f` with the liberated form of each one.
fn for_each_late_bound_region_defined_on<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
    mut f: impl FnMut(ty::Region<'tcx>),
) {
    if let Some(late_bounds) = tcx.is_late_bound_map(fn_def_id.expect_local()) {
        for late_bound in late_bounds.iter() {
            let hir_id = HirId { owner: fn_def_id.expect_local(), local_id: *late_bound };
            let name = tcx.hir().name(hir_id);
            let region_def_id = tcx.hir().local_def_id(hir_id);
            let liberated_region = tcx.mk_region(ty::ReFree(ty::FreeRegion {
                scope: fn_def_id,
                bound_region: ty::BoundRegion::BrNamed(region_def_id.to_def_id(), name),
            }));
            f(liberated_region);
        }
    }
}
