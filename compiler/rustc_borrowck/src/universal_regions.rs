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

#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::cell::Cell;
use std::iter;

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Diag;
use rustc_hir::BodyOwnerKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_macros::extension;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, GenericArgs, GenericArgsRef, InlineConstArgs, InlineConstArgsParts, RegionVid, Ty,
    TyCtxt, TypeFoldable, TypeVisitableExt, fold_regions,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{ErrorGuaranteed, kw, sym};
use tracing::{debug, instrument};

use crate::BorrowckInferCtxt;
use crate::renumber::RegionCtxt;

#[derive(Debug)]
#[derive(Clone)] // FIXME(#146079)
pub(crate) struct UniversalRegions<'tcx> {
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

    /// The "defining" type for this function, with all universal
    /// regions instantiated. For a closure or coroutine, this is the
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

    pub resume_ty: Option<Ty<'tcx>>,
}

/// The "defining type" for this MIR. The key feature of the "defining
/// type" is that it contains the information needed to derive all the
/// universal regions that are in scope as well as the types of the
/// inputs/output from the MIR. In general, early-bound universal
/// regions appear free in the defining type and late-bound regions
/// appear bound in the signature.
#[derive(Copy, Clone, Debug)]
pub(crate) enum DefiningTy<'tcx> {
    /// The MIR is a closure. The signature is found via
    /// `ClosureArgs::closure_sig_ty`.
    Closure(DefId, GenericArgsRef<'tcx>),

    /// The MIR is a coroutine. The signature is that coroutines take
    /// no parameters and return the result of
    /// `ClosureArgs::coroutine_return_ty`.
    Coroutine(DefId, GenericArgsRef<'tcx>),

    /// The MIR is a special kind of closure that returns coroutines.
    ///
    /// See the documentation on `CoroutineClosureSignature` for details
    /// on how to construct the callable signature of the coroutine from
    /// its args.
    CoroutineClosure(DefId, GenericArgsRef<'tcx>),

    /// The MIR is a fn item with the given `DefId` and args. The signature
    /// of the function can be bound then with the `fn_sig` query.
    FnDef(DefId, GenericArgsRef<'tcx>),

    /// The MIR represents some form of constant. The signature then
    /// is that it has no inputs and a single return value, which is
    /// the value of the constant.
    Const(DefId, GenericArgsRef<'tcx>),

    /// The MIR represents an inline const. The signature has no inputs and a
    /// single return value found via `InlineConstArgs::ty`.
    InlineConst(DefId, GenericArgsRef<'tcx>),

    // Fake body for a global asm. Not particularly useful or interesting,
    // but we need it so we can properly store the typeck results of the asm
    // operands, which aren't associated with a body otherwise.
    GlobalAsm(DefId),
}

impl<'tcx> DefiningTy<'tcx> {
    /// Returns a list of all the upvar types for this MIR. If this is
    /// not a closure or coroutine, there are no upvars, and hence it
    /// will be an empty list. The order of types in this list will
    /// match up with the upvar order in the HIR, typesystem, and MIR.
    pub(crate) fn upvar_tys(self) -> &'tcx ty::List<Ty<'tcx>> {
        match self {
            DefiningTy::Closure(_, args) => args.as_closure().upvar_tys(),
            DefiningTy::CoroutineClosure(_, args) => args.as_coroutine_closure().upvar_tys(),
            DefiningTy::Coroutine(_, args) => args.as_coroutine().upvar_tys(),
            DefiningTy::FnDef(..)
            | DefiningTy::Const(..)
            | DefiningTy::InlineConst(..)
            | DefiningTy::GlobalAsm(_) => ty::List::empty(),
        }
    }

    /// Number of implicit inputs -- notably the "environment"
    /// parameter for closures -- that appear in MIR but not in the
    /// user's code.
    pub(crate) fn implicit_inputs(self) -> usize {
        match self {
            DefiningTy::Closure(..)
            | DefiningTy::CoroutineClosure(..)
            | DefiningTy::Coroutine(..) => 1,
            DefiningTy::FnDef(..)
            | DefiningTy::Const(..)
            | DefiningTy::InlineConst(..)
            | DefiningTy::GlobalAsm(_) => 0,
        }
    }

    pub(crate) fn is_fn_def(&self) -> bool {
        matches!(*self, DefiningTy::FnDef(..))
    }

    pub(crate) fn is_const(&self) -> bool {
        matches!(*self, DefiningTy::Const(..) | DefiningTy::InlineConst(..))
    }

    pub(crate) fn def_id(&self) -> DefId {
        match *self {
            DefiningTy::Closure(def_id, ..)
            | DefiningTy::CoroutineClosure(def_id, ..)
            | DefiningTy::Coroutine(def_id, ..)
            | DefiningTy::FnDef(def_id, ..)
            | DefiningTy::Const(def_id, ..)
            | DefiningTy::InlineConst(def_id, ..)
            | DefiningTy::GlobalAsm(def_id) => def_id,
        }
    }

    /// Returns the args of the `DefiningTy`. These are equivalent to the identity
    /// substs of the body, but replaced with region vids.
    pub(crate) fn args(&self) -> ty::GenericArgsRef<'tcx> {
        match *self {
            DefiningTy::Closure(_, args)
            | DefiningTy::Coroutine(_, args)
            | DefiningTy::CoroutineClosure(_, args)
            | DefiningTy::FnDef(_, args)
            | DefiningTy::Const(_, args)
            | DefiningTy::InlineConst(_, args) => args,
            DefiningTy::GlobalAsm(_) => ty::List::empty(),
        }
    }
}

#[derive(Debug)]
#[derive(Clone)] // FIXME(#146079)
struct UniversalRegionIndices<'tcx> {
    /// For those regions that may appear in the parameter environment
    /// ('static and early-bound regions), we maintain a map from the
    /// `ty::Region` to the internal `RegionVid` we are using. This is
    /// used because trait matching and type-checking will feed us
    /// region constraints that reference those regions and we need to
    /// be able to map them to our internal `RegionVid`. This is
    /// basically equivalent to an `GenericArgs`, except that it also
    /// contains an entry for `ReStatic` -- it might be nice to just
    /// use an args, and then handle `ReStatic` another way.
    indices: FxIndexMap<ty::Region<'tcx>, RegionVid>,

    /// The vid assigned to `'static`. Used only for diagnostics.
    pub fr_static: RegionVid,

    /// Whether we've encountered an error region. If we have, cancel all
    /// outlives errors, as they are likely bogus.
    pub encountered_re_error: Cell<Option<ErrorGuaranteed>>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum RegionClassification {
    /// A **global** region is one that can be named from
    /// anywhere. There is only one, `'static`.
    Global,

    /// An **external** region is only relevant for
    /// closures, coroutines, and inline consts. In that
    /// case, it refers to regions that are free in the type
    /// -- basically, something bound in the surrounding context.
    ///
    /// Consider this example:
    ///
    /// ```ignore (pseudo-rust)
    /// fn foo<'a, 'b>(a: &'a u32, b: &'b u32, c: &'static u32) {
    ///   let closure = for<'x> |x: &'x u32| { .. };
    ///    //           ^^^^^^^ pretend this were legal syntax
    ///    //                   for declaring a late-bound region in
    ///    //                   a closure signature
    /// }
    /// ```
    ///
    /// Here, the lifetimes `'a` and `'b` would be **external** to the
    /// closure.
    ///
    /// If we are not analyzing a closure/coroutine/inline-const,
    /// there are no external lifetimes.
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
    pub(crate) fn new(infcx: &BorrowckInferCtxt<'tcx>, mir_def: LocalDefId) -> Self {
        UniversalRegionsBuilder { infcx, mir_def }.build()
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
    pub(crate) fn closure_mapping(
        tcx: TyCtxt<'tcx>,
        closure_args: GenericArgsRef<'tcx>,
        expected_num_vars: usize,
        closure_def_id: LocalDefId,
    ) -> IndexVec<RegionVid, ty::Region<'tcx>> {
        let mut region_mapping = IndexVec::with_capacity(expected_num_vars);
        region_mapping.push(tcx.lifetimes.re_static);
        tcx.for_each_free_region(&closure_args, |fr| {
            region_mapping.push(fr);
        });

        for_each_late_bound_region_in_recursive_scope(tcx, tcx.local_parent(closure_def_id), |r| {
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
    pub(crate) fn is_universal_region(&self, r: RegionVid) -> bool {
        (FIRST_GLOBAL_INDEX..self.num_universals).contains(&r.index())
    }

    /// Classifies `r` as a universal region, returning `None` if this
    /// is not a member of this set of universal regions.
    pub(crate) fn region_classification(&self, r: RegionVid) -> Option<RegionClassification> {
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
    pub(crate) fn universal_regions_iter(&self) -> impl Iterator<Item = RegionVid> + 'static {
        (FIRST_GLOBAL_INDEX..self.num_universals).map(RegionVid::from_usize)
    }

    /// Returns `true` if `r` is classified as a local region.
    pub(crate) fn is_local_free_region(&self, r: RegionVid) -> bool {
        self.region_classification(r) == Some(RegionClassification::Local)
    }

    /// Returns the number of universal regions created in any category.
    pub(crate) fn len(&self) -> usize {
        self.num_universals
    }

    /// Returns the number of global plus external universal regions.
    /// For closures, these are the regions that appear free in the
    /// closure type (versus those bound in the closure
    /// signature). They are therefore the regions between which the
    /// closure may impose constraints that its creator must verify.
    pub(crate) fn num_global_and_external_regions(&self) -> usize {
        self.first_local_index
    }

    /// Gets an iterator over all the early-bound regions that have names.
    pub(crate) fn named_universal_regions_iter(
        &self,
    ) -> impl Iterator<Item = (ty::Region<'tcx>, ty::RegionVid)> {
        self.indices.indices.iter().map(|(&r, &v)| (r, v))
    }

    /// See [UniversalRegionIndices::to_region_vid].
    pub(crate) fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.indices.to_region_vid(r)
    }

    /// As part of the NLL unit tests, you can annotate a function with
    /// `#[rustc_regions]`, and we will emit information about the region
    /// inference context and -- in particular -- the external constraints
    /// that this region imposes on others. The methods in this file
    /// handle the part about dumping the inference context internal
    /// state.
    pub(crate) fn annotate(&self, tcx: TyCtxt<'tcx>, err: &mut Diag<'_, ()>) {
        match self.defining_ty {
            DefiningTy::Closure(def_id, args) => {
                let v = with_no_trimmed_paths!(
                    args[tcx.generics_of(def_id).parent_count..]
                        .iter()
                        .map(|arg| arg.to_string())
                        .collect::<Vec<_>>()
                );
                err.note(format!(
                    "defining type: {} with closure args [\n    {},\n]",
                    tcx.def_path_str_with_args(def_id, args),
                    v.join(",\n    "),
                ));

                // FIXME: It'd be nice to print the late-bound regions
                // here, but unfortunately these wind up stored into
                // tests, and the resulting print-outs include def-ids
                // and other things that are not stable across tests!
                // So we just include the region-vid. Annoying.
                for_each_late_bound_region_in_recursive_scope(tcx, def_id.expect_local(), |r| {
                    err.note(format!("late-bound region is {:?}", self.to_region_vid(r)));
                });
            }
            DefiningTy::CoroutineClosure(..) => {
                todo!()
            }
            DefiningTy::Coroutine(def_id, args) => {
                let v = with_no_trimmed_paths!(
                    args[tcx.generics_of(def_id).parent_count..]
                        .iter()
                        .map(|arg| arg.to_string())
                        .collect::<Vec<_>>()
                );
                err.note(format!(
                    "defining type: {} with coroutine args [\n    {},\n]",
                    tcx.def_path_str_with_args(def_id, args),
                    v.join(",\n    "),
                ));

                // FIXME: As above, we'd like to print out the region
                // `r` but doing so is not stable across architectures
                // and so forth.
                for_each_late_bound_region_in_recursive_scope(tcx, def_id.expect_local(), |r| {
                    err.note(format!("late-bound region is {:?}", self.to_region_vid(r)));
                });
            }
            DefiningTy::FnDef(def_id, args) => {
                err.note(format!("defining type: {}", tcx.def_path_str_with_args(def_id, args),));
            }
            DefiningTy::Const(def_id, args) => {
                err.note(format!(
                    "defining constant type: {}",
                    tcx.def_path_str_with_args(def_id, args),
                ));
            }
            DefiningTy::InlineConst(def_id, args) => {
                err.note(format!(
                    "defining inline constant type: {}",
                    tcx.def_path_str_with_args(def_id, args),
                ));
            }
            DefiningTy::GlobalAsm(_) => unreachable!(),
        }
    }

    pub(crate) fn implicit_region_bound(&self) -> RegionVid {
        self.fr_fn_body
    }

    pub(crate) fn encountered_re_error(&self) -> Option<ErrorGuaranteed> {
        self.indices.encountered_re_error.get()
    }
}

struct UniversalRegionsBuilder<'infcx, 'tcx> {
    infcx: &'infcx BorrowckInferCtxt<'tcx>,
    mir_def: LocalDefId,
}

const FR: NllRegionVariableOrigin = NllRegionVariableOrigin::FreeRegion;

impl<'cx, 'tcx> UniversalRegionsBuilder<'cx, 'tcx> {
    fn build(self) -> UniversalRegions<'tcx> {
        debug!("build(mir_def={:?})", self.mir_def);

        let param_env = self.infcx.param_env;
        debug!("build: param_env={:?}", param_env);

        assert_eq!(FIRST_GLOBAL_INDEX, self.infcx.num_region_vars());

        // Create the "global" region that is always free in all contexts: 'static.
        let fr_static =
            self.infcx.next_nll_region_var(FR, || RegionCtxt::Free(kw::Static)).as_var();

        // We've now added all the global regions. The next ones we
        // add will be external.
        let first_extern_index = self.infcx.num_region_vars();

        let defining_ty = self.defining_ty();
        debug!("build: defining_ty={:?}", defining_ty);

        let mut indices = self.compute_indices(fr_static, defining_ty);
        debug!("build: indices={:?}", indices);

        let typeck_root_def_id = self.infcx.tcx.typeck_root_def_id(self.mir_def.to_def_id());

        // If this is a 'root' body (not a closure/coroutine/inline const), then
        // there are no extern regions, so the local regions start at the same
        // position as the (empty) sub-list of extern regions
        let first_local_index = if self.mir_def.to_def_id() == typeck_root_def_id {
            first_extern_index
        } else {
            // If this is a closure, coroutine, or inline-const, then the late-bound regions from the enclosing
            // function/closures are actually external regions to us. For example, here, 'a is not local
            // to the closure c (although it is local to the fn foo):
            // fn foo<'a>() {
            //     let c = || { let x: &'a u32 = ...; }
            // }
            for_each_late_bound_region_in_recursive_scope(
                self.infcx.tcx,
                self.infcx.tcx.local_parent(self.mir_def),
                |r| {
                    debug!(?r);
                    let region_vid = {
                        let name = r.get_name_or_anon(self.infcx.tcx);
                        self.infcx.next_nll_region_var(FR, || RegionCtxt::LateBound(name))
                    };

                    debug!(?region_vid);
                    indices.insert_late_bound_region(r, region_vid.as_var());
                },
            );

            // Any regions created during the execution of `defining_ty` or during the above
            // late-bound region replacement are all considered 'extern' regions
            self.infcx.num_region_vars()
        };

        // Converse of above, if this is a function/closure then the late-bound regions declared
        // on its signature are local.
        //
        // We manually loop over `bound_inputs_and_output` instead of using
        // `for_each_late_bound_region_in_item` as we may need to add the otherwise
        // implicit `ClosureEnv` region.
        let bound_inputs_and_output = self.compute_inputs_and_output(&indices, defining_ty);
        for (idx, bound_var) in bound_inputs_and_output.bound_vars().iter().enumerate() {
            if let ty::BoundVariableKind::Region(kind) = bound_var {
                let kind = ty::LateParamRegionKind::from_bound(ty::BoundVar::from_usize(idx), kind);
                let r = ty::Region::new_late_param(self.infcx.tcx, self.mir_def.to_def_id(), kind);
                let region_vid = {
                    let name = r.get_name_or_anon(self.infcx.tcx);
                    self.infcx.next_nll_region_var(FR, || RegionCtxt::LateBound(name))
                };

                debug!(?region_vid);
                indices.insert_late_bound_region(r, region_vid.as_var());
            }
        }
        let inputs_and_output = self.infcx.replace_bound_regions_with_nll_infer_vars(
            self.mir_def,
            bound_inputs_and_output,
            &indices,
        );

        let (unnormalized_output_ty, mut unnormalized_input_tys) =
            inputs_and_output.split_last().unwrap();

        // C-variadic fns also have a `VaList` input that's not listed in the signature
        // (as it's created inside the body itself, not passed in from outside).
        if let DefiningTy::FnDef(def_id, _) = defining_ty {
            if self.infcx.tcx.fn_sig(def_id).skip_binder().c_variadic() {
                let va_list_did = self
                    .infcx
                    .tcx
                    .require_lang_item(LangItem::VaList, self.infcx.tcx.def_span(self.mir_def));

                let reg_vid = self
                    .infcx
                    .next_nll_region_var(FR, || RegionCtxt::Free(sym::c_dash_variadic))
                    .as_var();

                let region = ty::Region::new_var(self.infcx.tcx, reg_vid);
                let va_list_ty = self
                    .infcx
                    .tcx
                    .type_of(va_list_did)
                    .instantiate(self.infcx.tcx, &[region.into()]);

                unnormalized_input_tys = self.infcx.tcx.mk_type_list_from_iter(
                    unnormalized_input_tys.iter().copied().chain(iter::once(va_list_ty)),
                );
            }
        }

        let fr_fn_body =
            self.infcx.next_nll_region_var(FR, || RegionCtxt::Free(sym::fn_body)).as_var();

        let num_universals = self.infcx.num_region_vars();

        debug!("build: global regions = {}..{}", FIRST_GLOBAL_INDEX, first_extern_index);
        debug!("build: extern regions = {}..{}", first_extern_index, first_local_index);
        debug!("build: local regions  = {}..{}", first_local_index, num_universals);

        let (resume_ty, yield_ty) = match defining_ty {
            DefiningTy::Coroutine(_, args) => {
                let tys = args.as_coroutine();
                (Some(tys.resume_ty()), Some(tys.yield_ty()))
            }
            _ => (None, None),
        };

        UniversalRegions {
            indices,
            fr_static,
            fr_fn_body,
            first_extern_index,
            first_local_index,
            num_universals,
            defining_ty,
            unnormalized_output_ty: *unnormalized_output_ty,
            unnormalized_input_tys,
            yield_ty,
            resume_ty,
        }
    }

    /// Returns the "defining type" of the current MIR;
    /// see `DefiningTy` for details.
    fn defining_ty(&self) -> DefiningTy<'tcx> {
        let tcx = self.infcx.tcx;
        let typeck_root_def_id = tcx.typeck_root_def_id(self.mir_def.to_def_id());

        match tcx.hir_body_owner_kind(self.mir_def) {
            BodyOwnerKind::Closure | BodyOwnerKind::Fn => {
                let defining_ty = tcx.type_of(self.mir_def).instantiate_identity();

                debug!("defining_ty (pre-replacement): {:?}", defining_ty);

                let defining_ty =
                    self.infcx.replace_free_regions_with_nll_infer_vars(FR, defining_ty);

                match *defining_ty.kind() {
                    ty::Closure(def_id, args) => DefiningTy::Closure(def_id, args),
                    ty::Coroutine(def_id, args) => DefiningTy::Coroutine(def_id, args),
                    ty::CoroutineClosure(def_id, args) => {
                        DefiningTy::CoroutineClosure(def_id, args)
                    }
                    ty::FnDef(def_id, args) => DefiningTy::FnDef(def_id, args),
                    _ => span_bug!(
                        tcx.def_span(self.mir_def),
                        "expected defining type for `{:?}`: `{:?}`",
                        self.mir_def,
                        defining_ty
                    ),
                }
            }

            BodyOwnerKind::Const { .. } | BodyOwnerKind::Static(..) => {
                let identity_args = GenericArgs::identity_for_item(tcx, typeck_root_def_id);
                if self.mir_def.to_def_id() == typeck_root_def_id
                    // Do not ICE when checking default_field_values consts with lifetimes (#135649)
                    && DefKind::Field != tcx.def_kind(tcx.parent(typeck_root_def_id))
                {
                    let args =
                        self.infcx.replace_free_regions_with_nll_infer_vars(FR, identity_args);
                    DefiningTy::Const(self.mir_def.to_def_id(), args)
                } else {
                    // FIXME this line creates a dependency between borrowck and typeck.
                    //
                    // This is required for `AscribeUserType` canonical query, which will call
                    // `type_of(inline_const_def_id)`. That `type_of` would inject erased lifetimes
                    // into borrowck, which is ICE #78174.
                    //
                    // As a workaround, inline consts have an additional generic param (`ty`
                    // below), so that `type_of(inline_const_def_id).args(args)` uses the
                    // proper type with NLL infer vars.
                    let ty = tcx
                        .typeck(self.mir_def)
                        .node_type(tcx.local_def_id_to_hir_id(self.mir_def));
                    let args = InlineConstArgs::new(
                        tcx,
                        InlineConstArgsParts { parent_args: identity_args, ty },
                    )
                    .args;
                    let args = self.infcx.replace_free_regions_with_nll_infer_vars(FR, args);
                    DefiningTy::InlineConst(self.mir_def.to_def_id(), args)
                }
            }

            BodyOwnerKind::GlobalAsm => DefiningTy::GlobalAsm(self.mir_def.to_def_id()),
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
        let typeck_root_def_id = tcx.typeck_root_def_id(self.mir_def.to_def_id());
        let identity_args = GenericArgs::identity_for_item(tcx, typeck_root_def_id);
        let fr_args = match defining_ty {
            DefiningTy::Closure(_, args)
            | DefiningTy::CoroutineClosure(_, args)
            | DefiningTy::Coroutine(_, args)
            | DefiningTy::InlineConst(_, args) => {
                // In the case of closures, we rely on the fact that
                // the first N elements in the ClosureArgs are
                // inherited from the `typeck_root_def_id`.
                // Therefore, when we zip together (below) with
                // `identity_args`, we will get only those regions
                // that correspond to early-bound regions declared on
                // the `typeck_root_def_id`.
                assert!(args.len() >= identity_args.len());
                assert_eq!(args.regions().count(), identity_args.regions().count());
                args
            }

            DefiningTy::FnDef(_, args) | DefiningTy::Const(_, args) => args,

            DefiningTy::GlobalAsm(_) => ty::List::empty(),
        };

        let global_mapping = iter::once((tcx.lifetimes.re_static, fr_static));
        let arg_mapping = iter::zip(identity_args.regions(), fr_args.regions().map(|r| r.as_var()));

        UniversalRegionIndices {
            indices: global_mapping.chain(arg_mapping).collect(),
            fr_static,
            encountered_re_error: Cell::new(None),
        }
    }

    fn compute_inputs_and_output(
        &self,
        indices: &UniversalRegionIndices<'tcx>,
        defining_ty: DefiningTy<'tcx>,
    ) -> ty::Binder<'tcx, &'tcx ty::List<Ty<'tcx>>> {
        let tcx = self.infcx.tcx;

        let inputs_and_output = match defining_ty {
            DefiningTy::Closure(def_id, args) => {
                assert_eq!(self.mir_def.to_def_id(), def_id);
                let closure_sig = args.as_closure().sig();
                let inputs_and_output = closure_sig.inputs_and_output();
                let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                    inputs_and_output.bound_vars().iter().chain(iter::once(
                        ty::BoundVariableKind::Region(ty::BoundRegionKind::ClosureEnv),
                    )),
                );
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                    kind: ty::BoundRegionKind::ClosureEnv,
                };
                let env_region = ty::Region::new_bound(tcx, ty::INNERMOST, br);
                let closure_ty = tcx.closure_env_ty(
                    Ty::new_closure(tcx, def_id, args),
                    args.as_closure().kind(),
                    env_region,
                );

                // The "inputs" of the closure in the
                // signature appear as a tuple. The MIR side
                // flattens this tuple.
                let (&output, tuplized_inputs) =
                    inputs_and_output.skip_binder().split_last().unwrap();
                assert_eq!(tuplized_inputs.len(), 1, "multiple closure inputs");
                let &ty::Tuple(inputs) = tuplized_inputs[0].kind() else {
                    bug!("closure inputs not a tuple: {:?}", tuplized_inputs[0]);
                };

                ty::Binder::bind_with_vars(
                    tcx.mk_type_list_from_iter(
                        iter::once(closure_ty).chain(inputs).chain(iter::once(output)),
                    ),
                    bound_vars,
                )
            }

            DefiningTy::Coroutine(def_id, args) => {
                assert_eq!(self.mir_def.to_def_id(), def_id);
                let resume_ty = args.as_coroutine().resume_ty();
                let output = args.as_coroutine().return_ty();
                let coroutine_ty = Ty::new_coroutine(tcx, def_id, args);
                let inputs_and_output =
                    self.infcx.tcx.mk_type_list(&[coroutine_ty, resume_ty, output]);
                ty::Binder::dummy(inputs_and_output)
            }

            // Construct the signature of the CoroutineClosure for the purposes of borrowck.
            // This is pretty straightforward -- we:
            // 1. first grab the `coroutine_closure_sig`,
            // 2. compute the self type (`&`/`&mut`/no borrow),
            // 3. flatten the tupled_input_tys,
            // 4. construct the correct generator type to return with
            //    `CoroutineClosureSignature::to_coroutine_given_kind_and_upvars`.
            // Then we wrap it all up into a list of inputs and output.
            DefiningTy::CoroutineClosure(def_id, args) => {
                assert_eq!(self.mir_def.to_def_id(), def_id);
                let closure_sig = args.as_coroutine_closure().coroutine_closure_sig();
                let bound_vars =
                    tcx.mk_bound_variable_kinds_from_iter(closure_sig.bound_vars().iter().chain(
                        iter::once(ty::BoundVariableKind::Region(ty::BoundRegionKind::ClosureEnv)),
                    ));
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                    kind: ty::BoundRegionKind::ClosureEnv,
                };
                let env_region = ty::Region::new_bound(tcx, ty::INNERMOST, br);
                let closure_kind = args.as_coroutine_closure().kind();

                let closure_ty = tcx.closure_env_ty(
                    Ty::new_coroutine_closure(tcx, def_id, args),
                    closure_kind,
                    env_region,
                );

                let inputs = closure_sig.skip_binder().tupled_inputs_ty.tuple_fields();
                let output = closure_sig.skip_binder().to_coroutine_given_kind_and_upvars(
                    tcx,
                    args.as_coroutine_closure().parent_args(),
                    tcx.coroutine_for_closure(def_id),
                    closure_kind,
                    env_region,
                    args.as_coroutine_closure().tupled_upvars_ty(),
                    args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
                );

                ty::Binder::bind_with_vars(
                    tcx.mk_type_list_from_iter(
                        iter::once(closure_ty).chain(inputs).chain(iter::once(output)),
                    ),
                    bound_vars,
                )
            }

            DefiningTy::FnDef(def_id, _) => {
                let sig = tcx.fn_sig(def_id).instantiate_identity();
                let sig = indices.fold_to_region_vids(tcx, sig);
                sig.inputs_and_output()
            }

            DefiningTy::Const(def_id, _) => {
                // For a constant body, there are no inputs, and one
                // "output" (the type of the constant).
                assert_eq!(self.mir_def.to_def_id(), def_id);
                let ty = tcx.type_of(self.mir_def).instantiate_identity();

                let ty = indices.fold_to_region_vids(tcx, ty);
                ty::Binder::dummy(tcx.mk_type_list(&[ty]))
            }

            DefiningTy::InlineConst(def_id, args) => {
                assert_eq!(self.mir_def.to_def_id(), def_id);
                let ty = args.as_inline_const().ty();
                ty::Binder::dummy(tcx.mk_type_list(&[ty]))
            }

            DefiningTy::GlobalAsm(def_id) => {
                ty::Binder::dummy(tcx.mk_type_list(&[tcx.type_of(def_id).instantiate_identity()]))
            }
        };

        // FIXME(#129952): We probably want a more principled approach here.
        if let Err(terr) = inputs_and_output.skip_binder().error_reported() {
            self.infcx.set_tainted_by_errors(terr);
        }

        inputs_and_output
    }
}

#[extension(trait InferCtxtExt<'tcx>)]
impl<'tcx> BorrowckInferCtxt<'tcx> {
    #[instrument(skip(self), level = "debug")]
    fn replace_free_regions_with_nll_infer_vars<T>(
        &self,
        origin: NllRegionVariableOrigin,
        value: T,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(self.infcx.tcx, value, |region, _depth| {
            let name = region.get_name_or_anon(self.infcx.tcx);
            debug!(?region, ?name);

            self.next_nll_region_var(origin, || RegionCtxt::Free(name))
        })
    }

    #[instrument(level = "debug", skip(self, indices))]
    fn replace_bound_regions_with_nll_infer_vars<T>(
        &self,
        all_outlive_scope: LocalDefId,
        value: ty::Binder<'tcx, T>,
        indices: &UniversalRegionIndices<'tcx>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (value, _map) = self.tcx.instantiate_bound_regions(value, |br| {
            debug!(?br);
            let kind = ty::LateParamRegionKind::from_bound(br.var, br.kind);
            let liberated_region =
                ty::Region::new_late_param(self.tcx, all_outlive_scope.to_def_id(), kind);
            ty::Region::new_var(self.tcx, indices.to_region_vid(liberated_region))
        });
        value
    }
}

impl<'tcx> UniversalRegionIndices<'tcx> {
    /// Initially, the `UniversalRegionIndices` map contains only the
    /// early-bound regions in scope. Once that is all setup, we come
    /// in later and instantiate the late-bound regions, and then we
    /// insert the `ReLateParam` version of those into the map as
    /// well. These are used for error reporting.
    fn insert_late_bound_region(&mut self, r: ty::Region<'tcx>, vid: ty::RegionVid) {
        debug!("insert_late_bound_region({:?}, {:?})", r, vid);
        assert_eq!(self.indices.insert(r, vid), None);
    }

    /// Converts `r` into a local inference variable: `r` can either
    /// be a `ReVar` (i.e., already a reference to an inference
    /// variable) or it can be `'static` or some early-bound
    /// region. This is useful when taking the results from
    /// type-checking and trait-matching, which may sometimes
    /// reference those regions from the `ParamEnv`. It is also used
    /// during initialization. Relies on the `indices` map having been
    /// fully initialized.
    ///
    /// Panics if `r` is not a registered universal region, most notably
    /// if it is a placeholder. Handling placeholders requires access to the
    /// `MirTypeckRegionConstraints`.
    fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        match r.kind() {
            ty::ReVar(..) => r.as_var(),
            ty::ReError(guar) => {
                self.encountered_re_error.set(Some(guar));
                // We use the `'static` `RegionVid` because `ReError` doesn't actually exist in the
                // `UniversalRegionIndices`. This is fine because 1) it is a fallback only used if
                // errors are being emitted and 2) it leaves the happy path unaffected.
                self.fr_static
            }
            _ => *self
                .indices
                .get(&r)
                .unwrap_or_else(|| bug!("cannot convert `{:?}` to a region vid", r)),
        }
    }

    /// Replaces all free regions in `value` with region vids, as
    /// returned by `to_region_vid`.
    fn fold_to_region_vids<T>(&self, tcx: TyCtxt<'tcx>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(tcx, value, |region, _| ty::Region::new_var(tcx, self.to_region_vid(region)))
    }
}

/// Iterates over the late-bound regions defined on `mir_def_id` and all of its
/// parents, up to the typeck root, and invokes `f` with the liberated form
/// of each one.
fn for_each_late_bound_region_in_recursive_scope<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut mir_def_id: LocalDefId,
    mut f: impl FnMut(ty::Region<'tcx>),
) {
    let typeck_root_def_id = tcx.typeck_root_def_id(mir_def_id.to_def_id());

    // Walk up the tree, collecting late-bound regions until we hit the typeck root
    loop {
        for_each_late_bound_region_in_item(tcx, mir_def_id, &mut f);

        if mir_def_id.to_def_id() == typeck_root_def_id {
            break;
        } else {
            mir_def_id = tcx.local_parent(mir_def_id);
        }
    }
}

/// Iterates over the late-bound regions defined on `mir_def_id` and all of its
/// parents, up to the typeck root, and invokes `f` with the liberated form
/// of each one.
fn for_each_late_bound_region_in_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_def_id: LocalDefId,
    mut f: impl FnMut(ty::Region<'tcx>),
) {
    let bound_vars = match tcx.def_kind(mir_def_id) {
        DefKind::Fn | DefKind::AssocFn => {
            tcx.late_bound_vars(tcx.local_def_id_to_hir_id(mir_def_id))
        }
        // We extract the bound vars from the deduced closure signature, since we may have
        // only deduced that a param in the closure signature is late-bound from a constraint
        // that we discover during typeck.
        DefKind::Closure => {
            let ty = tcx.type_of(mir_def_id).instantiate_identity();
            match *ty.kind() {
                ty::Closure(_, args) => args.as_closure().sig().bound_vars(),
                ty::CoroutineClosure(_, args) => {
                    args.as_coroutine_closure().coroutine_closure_sig().bound_vars()
                }
                ty::Coroutine(_, _) | ty::Error(_) => return,
                _ => unreachable!("unexpected type for closure: {ty}"),
            }
        }
        _ => return,
    };

    for (idx, bound_var) in bound_vars.iter().enumerate() {
        if let ty::BoundVariableKind::Region(kind) = bound_var {
            let kind = ty::LateParamRegionKind::from_bound(ty::BoundVar::from_usize(idx), kind);
            let liberated_region = ty::Region::new_late_param(tcx, mir_def_id.to_def_id(), kind);
            f(liberated_region);
        }
    }
}
