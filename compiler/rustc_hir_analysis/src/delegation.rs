//! Support inheriting generic parameters and predicates for function delegation.
//!
//! For more information about delegation design, see the tracking issue #118212.

use std::debug_assert_matches;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{HirId, PathSegment};
use rustc_middle::ty::{
    self, EarlyBinder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};
use rustc_span::{ErrorGuaranteed, Span, kw};

use crate::collect::ItemCtxt;
use crate::hir_ty_lowering::HirTyLowerer;

type RemapTable = FxHashMap<u32, u32>;

struct ParamIndexRemapper<'tcx> {
    tcx: TyCtxt<'tcx>,
    remap_table: RemapTable,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ParamIndexRemapper<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_param() {
            return ty;
        }

        if let ty::Param(param) = ty.kind()
            && let Some(index) = self.remap_table.get(&param.index)
        {
            return Ty::new_param(self.tcx, *index, param.name);
        }
        ty.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReEarlyParam(param) = r.kind()
            && let Some(index) = self.remap_table.get(&param.index).copied()
        {
            return ty::Region::new_early_param(
                self.tcx,
                ty::EarlyParamRegion { index, name: param.name },
            );
        }
        r
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Param(param) = ct.kind()
            && let Some(idx) = self.remap_table.get(&param.index)
        {
            let param = ty::ParamConst::new(*idx, param.name);
            return ty::Const::new_param(self.tcx, param);
        }
        ct.super_fold_with(self)
    }
}

enum SelfPositionKind {
    AfterLifetimes,
    Zero,
    None,
}

fn create_self_position_kind(caller_kind: FnKind, callee_kind: FnKind) -> SelfPositionKind {
    match (caller_kind, callee_kind) {
        (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::Free) => SelfPositionKind::Zero,

        (FnKind::Free, FnKind::AssocTrait) => SelfPositionKind::AfterLifetimes,

        _ => SelfPositionKind::None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FnKind {
    Free,
    AssocInherentImpl,
    AssocTrait,
    AssocTraitImpl,
}

fn fn_kind<'tcx>(tcx: TyCtxt<'tcx>, def_id: impl Into<DefId>) -> FnKind {
    let def_id = def_id.into();

    debug_assert_matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn);

    let parent = tcx.parent(def_id);
    match tcx.def_kind(parent) {
        DefKind::Trait => FnKind::AssocTrait,
        DefKind::Impl { of_trait: true } => FnKind::AssocTraitImpl,
        DefKind::Impl { of_trait: false } => FnKind::AssocInherentImpl,
        _ => FnKind::Free,
    }
}

/// Given the current context(caller and callee `FnKind`), it specifies
/// the policy of predicates and generic parameters inheritance.
#[derive(Clone, Copy, Debug, PartialEq)]
enum InheritanceKind {
    /// Copying all predicates and parameters, including those of the parent
    /// container.
    ///
    /// Boolean value defines whether the `Self` parameter or `Self: Trait`
    /// predicate are copied. It's always equal to `false` except when
    /// delegating from a free function to a trait method.
    ///
    /// FIXME(fn_delegation): This often leads to type inference
    /// errors. Support providing generic arguments or restrict use sites.
    WithParent(bool),
    /// The trait implementation should be compatible with the original trait.
    /// Therefore, for trait implementations only the method's own parameters
    /// and predicates are copied.
    Own,
}

/// Maps sig generics into generic args of delegation. Delegation generics has the following pattern:
///
/// [SELF | maybe self in the beginning]
/// [PARENT | args of delegation parent]
/// [SIG PARENT LIFETIMES]
/// [SIG LIFETIMES]
/// [SELF | maybe self after lifetimes, when we reuse trait fn in free context]
/// [SIG PARENT TYPES/CONSTS]
/// [SIG TYPES/CONSTS]
fn create_mapping<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    def_id: LocalDefId,
) -> FxHashMap<u32, u32> {
    let mut mapping: FxHashMap<u32, u32> = Default::default();

    let (caller_kind, callee_kind) = (fn_kind(tcx, def_id), fn_kind(tcx, sig_id));
    let self_pos_kind = create_self_position_kind(caller_kind, callee_kind);
    let is_self_at_zero = matches!(self_pos_kind, SelfPositionKind::Zero);

    // Is self at zero? If so insert mapping, self in sig parent is always at 0.
    if is_self_at_zero {
        mapping.insert(0, 0);
    }

    let mut args_index = 0;

    args_index += is_self_at_zero as usize;
    args_index += get_delegation_parent_args_count_without_self(tcx, def_id, sig_id);

    let sig_generics = tcx.generics_of(sig_id);
    let process_sig_parent_generics = matches!(callee_kind, FnKind::AssocTrait);

    if process_sig_parent_generics {
        for i in (sig_generics.has_self as usize)..sig_generics.parent_count {
            let param = sig_generics.param_at(i, tcx);
            if !param.kind.is_ty_or_const() {
                mapping.insert(param.index, args_index as u32);
                args_index += 1;
            }
        }
    }

    for param in &sig_generics.own_params {
        if !param.kind.is_ty_or_const() {
            mapping.insert(param.index, args_index as u32);
            args_index += 1;
        }
    }

    // If self after lifetimes insert mapping, relying that self is at 0 in sig parent.
    if matches!(self_pos_kind, SelfPositionKind::AfterLifetimes) {
        mapping.insert(0, args_index as u32);
        args_index += 1;
    }

    if process_sig_parent_generics {
        for i in (sig_generics.has_self as usize)..sig_generics.parent_count {
            let param = sig_generics.param_at(i, tcx);
            if param.kind.is_ty_or_const() {
                mapping.insert(param.index, args_index as u32);
                args_index += 1;
            }
        }
    }

    for param in &sig_generics.own_params {
        if param.kind.is_ty_or_const() {
            mapping.insert(param.index, args_index as u32);
            args_index += 1;
        }
    }

    mapping
}

fn get_delegation_parent_args_count_without_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    delegation_id: LocalDefId,
    sig_id: DefId,
) -> usize {
    let delegation_parent_args_count = tcx.generics_of(delegation_id).parent_count;

    match (fn_kind(tcx, delegation_id), fn_kind(tcx, sig_id)) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait)
        | (FnKind::AssocTraitImpl, FnKind::AssocTrait) => 0,

        (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            delegation_parent_args_count /* No Self in AssocInherentImpl */
        }

        (FnKind::AssocTrait, FnKind::Free) | (FnKind::AssocTrait, FnKind::AssocTrait) => {
            delegation_parent_args_count - 1 /* Without Self */
        }

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

fn get_parent_and_inheritance_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> (Option<DefId>, InheritanceKind) {
    match (fn_kind(tcx, def_id), fn_kind(tcx, sig_id)) {
        (FnKind::Free, FnKind::Free) | (FnKind::Free, FnKind::AssocTrait) => {
            (None, InheritanceKind::WithParent(true))
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait) => {
            (Some(tcx.parent(def_id.to_def_id())), InheritanceKind::Own)
        }

        (FnKind::AssocInherentImpl, FnKind::AssocTrait)
        | (FnKind::AssocTrait, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free) => {
            (Some(tcx.parent(def_id.to_def_id())), InheritanceKind::WithParent(false))
        }

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

fn get_delegation_self_ty<'tcx>(tcx: TyCtxt<'tcx>, delegation_id: LocalDefId) -> Option<Ty<'tcx>> {
    let sig_id = tcx.hir_opt_delegation_sig_id(delegation_id).expect("Delegation must have sig_id");
    let (caller_kind, callee_kind) = (fn_kind(tcx, delegation_id), fn_kind(tcx, sig_id));

    match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::Free, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::AssocTrait) => {
            match create_self_position_kind(caller_kind, callee_kind) {
                SelfPositionKind::None => None,
                SelfPositionKind::AfterLifetimes => {
                    // Both sig parent and child lifetimes are in included in this count.
                    Some(tcx.generics_of(delegation_id).own_counts().lifetimes)
                }
                SelfPositionKind::Zero => Some(0),
            }
            .map(|self_index| Ty::new_param(tcx, self_index as u32, kw::SelfUpper))
        }

        (FnKind::AssocTraitImpl, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            Some(tcx.type_of(tcx.local_parent(delegation_id)).instantiate_identity())
        }

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    }
}

/// Creates generic arguments for further delegation signature and predicates instantiation.
/// Arguments can be user-specified (in this case they are in `parent_args` and `child_args`)
/// or propagated. User can specify either both `parent_args` and `child_args`, one of them or none,
/// that is why we firstly create generic arguments from generic params and then adjust them with
/// user-specified args.
///
/// The order of produced list is important, it must be of this pattern:
///
/// [SELF | maybe self in the beginning]
/// [PARENT | args of delegation parent]
/// [SIG PARENT LIFETIMES] <- `lifetimes_end_pos`
/// [SIG LIFETIMES]
/// [SELF | maybe self after lifetimes, when we reuse trait fn in free context]
/// [SIG PARENT TYPES/CONSTS]
/// [SIG TYPES/CONSTS]
fn create_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig_id: DefId,
    delegation_id: LocalDefId,
    mut parent_args: &[ty::GenericArg<'tcx>],
    child_args: &[ty::GenericArg<'tcx>],
) -> Vec<ty::GenericArg<'tcx>> {
    let (caller_kind, callee_kind) = (fn_kind(tcx, delegation_id), fn_kind(tcx, sig_id));

    let delegation_args = ty::GenericArgs::identity_for_item(tcx, delegation_id);
    let delegation_parent_args_count = tcx.generics_of(delegation_id).parent_count;

    let deleg_parent_args_without_self_count =
        get_delegation_parent_args_count_without_self(tcx, delegation_id, sig_id);

    let args = match (caller_kind, callee_kind) {
        (FnKind::Free, FnKind::Free)
        | (FnKind::Free, FnKind::AssocTrait)
        | (FnKind::AssocInherentImpl, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::Free)
        | (FnKind::AssocTrait, FnKind::AssocTrait) => delegation_args,

        (FnKind::AssocTraitImpl, FnKind::AssocTrait) => {
            // Special case, as user specifies Trait args in trait impl header, we want to treat
            // them as parent args. We always generate a function whose generics match
            // child generics in trait.
            let parent = tcx.local_parent(delegation_id);
            parent_args = tcx.impl_trait_header(parent).trait_ref.instantiate_identity().args;

            assert!(child_args.is_empty(), "Child args can not be used in trait impl case");

            tcx.mk_args(&delegation_args[delegation_parent_args_count..])
        }

        (FnKind::AssocInherentImpl, FnKind::AssocTrait) => {
            let self_ty = tcx.type_of(tcx.local_parent(delegation_id)).instantiate_identity();

            tcx.mk_args_from_iter(
                std::iter::once(ty::GenericArg::from(self_ty)).chain(delegation_args.iter()),
            )
        }

        // For trait impl's `sig_id` is always equal to the corresponding trait method.
        // For inherent methods delegation is not yet supported.
        (FnKind::AssocTraitImpl, _)
        | (_, FnKind::AssocTraitImpl)
        | (_, FnKind::AssocInherentImpl) => unreachable!(),
    };

    let mut new_args = vec![];

    let self_pos_kind = create_self_position_kind(caller_kind, callee_kind);
    let mut lifetimes_end_pos;

    if !parent_args.is_empty() {
        let parent_args_lifetimes_count =
            parent_args.iter().filter(|a| a.as_region().is_some()).count();

        match self_pos_kind {
            SelfPositionKind::AfterLifetimes => {
                new_args.extend(&parent_args[1..1 + parent_args_lifetimes_count]);

                lifetimes_end_pos = parent_args_lifetimes_count;

                new_args.push(parent_args[0]);

                new_args.extend(&parent_args[1 + parent_args_lifetimes_count..]);
            }
            SelfPositionKind::Zero => {
                lifetimes_end_pos = 1 /* Self */ + parent_args_lifetimes_count;
                new_args.extend_from_slice(parent_args);

                for i in 0..deleg_parent_args_without_self_count {
                    new_args.insert(1 + i, args[1 + i]);
                }

                lifetimes_end_pos += deleg_parent_args_without_self_count;
            }
            // If we have parent args then we obtained them from trait, then self must be somewhere
            SelfPositionKind::None => unreachable!(),
        };
    } else {
        let self_impact = matches!(self_pos_kind, SelfPositionKind::Zero) as usize;

        lifetimes_end_pos = self_impact
            + deleg_parent_args_without_self_count
            + &args[self_impact + deleg_parent_args_without_self_count..]
                .iter()
                .filter(|a| a.as_region().is_some())
                .count();

        new_args.extend_from_slice(args);
    }

    if !child_args.is_empty() {
        let child_lifetimes_count = child_args.iter().filter(|a| a.as_region().is_some()).count();

        for i in 0..child_lifetimes_count {
            new_args.insert(lifetimes_end_pos + i, child_args[i]);
        }

        new_args.extend_from_slice(&child_args[child_lifetimes_count..]);
    } else if !parent_args.is_empty() {
        let child_args = &delegation_args[delegation_parent_args_count..];

        let child_lifetimes_count =
            child_args.iter().take_while(|a| a.as_region().is_some()).count();

        for i in 0..child_lifetimes_count {
            new_args.insert(lifetimes_end_pos + i, child_args[i]);
        }

        let skip_self = matches!(self_pos_kind, SelfPositionKind::AfterLifetimes);
        new_args.extend(&child_args[child_lifetimes_count + skip_self as usize..]);
    }

    new_args
}

pub(crate) fn inherit_predicates_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> ty::GenericPredicates<'tcx> {
    struct PredicatesCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: Vec<(ty::Clause<'tcx>, Span)>,
        args: Vec<ty::GenericArg<'tcx>>,
        folder: ParamIndexRemapper<'tcx>,
    }

    impl<'tcx> PredicatesCollector<'tcx> {
        fn with_own_preds(
            mut self,
            f: impl Fn(DefId) -> ty::GenericPredicates<'tcx>,
            def_id: DefId,
        ) -> Self {
            let preds = f(def_id);
            let args = self.args.as_slice();

            for pred in preds.predicates {
                let new_pred = pred.0.fold_with(&mut self.folder);
                self.preds.push((EarlyBinder::bind(new_pred).instantiate(self.tcx, args), pred.1));
            }

            self
        }

        fn with_preds(
            mut self,
            f: impl Fn(DefId) -> ty::GenericPredicates<'tcx> + Copy,
            def_id: DefId,
        ) -> Self {
            let preds = f(def_id);
            if let Some(parent_def_id) = preds.parent {
                self = self.with_own_preds(f, parent_def_id);
            }

            self.with_own_preds(f, def_id)
        }
    }

    let (parent_args, child_args) = get_delegation_user_specified_args(tcx, def_id);
    let (folder, args) = create_folder_and_args(tcx, def_id, sig_id, parent_args, child_args);
    let collector = PredicatesCollector { tcx, preds: vec![], args, folder };

    let (parent, inh_kind) = get_parent_and_inheritance_kind(tcx, def_id, sig_id);

    // `explicit_predicates_of` is used here to avoid copying `Self: Trait` predicate.
    // Note: `predicates_of` query can also add inferred outlives predicates, but that
    // is not the case here as `sig_id` is either a trait or a function.
    let preds = match inh_kind {
        InheritanceKind::WithParent(false) => {
            collector.with_preds(|def_id| tcx.explicit_predicates_of(def_id), sig_id)
        }
        InheritanceKind::WithParent(true) => {
            collector.with_preds(|def_id| tcx.predicates_of(def_id), sig_id)
        }
        InheritanceKind::Own => {
            collector.with_own_preds(|def_id| tcx.predicates_of(def_id), sig_id)
        }
    }
    .preds;

    ty::GenericPredicates { parent, predicates: tcx.arena.alloc_from_iter(preds) }
}

fn create_folder_and_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
    parent_args: &'tcx [ty::GenericArg<'tcx>],
    child_args: &'tcx [ty::GenericArg<'tcx>],
) -> (ParamIndexRemapper<'tcx>, Vec<ty::GenericArg<'tcx>>) {
    let args = create_generic_args(tcx, sig_id, def_id, parent_args, child_args);
    let remap_table = create_mapping(tcx, sig_id, def_id);

    (ParamIndexRemapper { tcx, remap_table }, args)
}

fn check_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    sig_id: DefId,
) -> Result<(), ErrorGuaranteed> {
    let mut ret = Ok(());

    let mut emit = |descr| {
        ret = Err(tcx.dcx().emit_err(crate::errors::UnsupportedDelegation {
            span: tcx.def_span(def_id),
            descr,
            callee_span: tcx.def_span(sig_id),
        }));
    };

    if tcx.fn_sig(sig_id).skip_binder().skip_binder().c_variadic() {
        // See issue #127443 for explanation.
        emit("delegation to C-variadic functions is not allowed");
    }

    ret
}

pub(crate) fn inherit_sig_for_delegation_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [Ty<'tcx>] {
    let sig_id = tcx.hir_opt_delegation_sig_id(def_id).expect("Delegation must have sig_id");
    let caller_sig = tcx.fn_sig(sig_id);

    if let Err(err) = check_constraints(tcx, def_id, sig_id) {
        let sig_len = caller_sig.instantiate_identity().skip_binder().inputs().len() + 1;
        let err_type = Ty::new_error(tcx, err);
        return tcx.arena.alloc_from_iter((0..sig_len).map(|_| err_type));
    }

    let (parent_args, child_args) = get_delegation_user_specified_args(tcx, def_id);
    let (mut folder, args) = create_folder_and_args(tcx, def_id, sig_id, parent_args, child_args);
    let caller_sig = EarlyBinder::bind(caller_sig.skip_binder().fold_with(&mut folder));

    let sig = caller_sig.instantiate(tcx, args.as_slice()).skip_binder();
    let sig_iter = sig.inputs().iter().cloned().chain(std::iter::once(sig.output()));
    tcx.arena.alloc_from_iter(sig_iter)
}

// Creates user-specified generic arguments from delegation path,
// they will be used during delegation signature and predicates inheritance.
// Example: reuse Trait::<'static, i32, 1>::foo::<A, B>
// we want to extract [Self, 'static, i32, 1] for parent and [A, B] for child.
fn get_delegation_user_specified_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    delegation_id: LocalDefId,
) -> (&'tcx [ty::GenericArg<'tcx>], &'tcx [ty::GenericArg<'tcx>]) {
    let info = tcx
        .hir_node(tcx.local_def_id_to_hir_id(delegation_id))
        .fn_sig()
        .expect("Lowering delegation")
        .decl
        .opt_delegation_generics()
        .expect("Lowering delegation");

    let get_segment = |hir_id: HirId| -> Option<(&'tcx PathSegment<'tcx>, DefId)> {
        let segment = tcx.hir_node(hir_id).expect_path_segment();
        segment.res.opt_def_id().map(|def_id| (segment, def_id))
    };

    let ctx = ItemCtxt::new(tcx, delegation_id);
    let lowerer = ctx.lowerer();

    let parent_args = info.parent_args_segment_id.and_then(get_segment).map(|(segment, def_id)| {
        let self_ty = get_delegation_self_ty(tcx, delegation_id);

        lowerer
            .lower_generic_args_of_path(segment.ident.span, def_id, &[], segment, self_ty)
            .0
            .as_slice()
    });

    let child_args = info
        .child_args_segment_id
        .and_then(get_segment)
        .filter(|(_, def_id)| matches!(tcx.def_kind(*def_id), DefKind::Fn | DefKind::AssocFn))
        .map(|(segment, def_id)| {
            let parent_args = if let Some(parent_args) = parent_args {
                parent_args
            } else {
                let parent = tcx.parent(def_id);
                if matches!(tcx.def_kind(parent), DefKind::Trait) {
                    ty::GenericArgs::identity_for_item(tcx, parent).as_slice()
                } else {
                    &[]
                }
            };

            let args = lowerer
                .lower_generic_args_of_path(segment.ident.span, def_id, parent_args, segment, None)
                .0;

            &args[parent_args.len()..]
        });

    (parent_args.unwrap_or_default(), child_args.unwrap_or_default())
}
