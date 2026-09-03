use std::ops::ControlFlow;

use ast::visit::Visitor;
use hir::def::DefKind;
use rustc_ast::{
    self as ast, AssocItemKind, AstOwner, Delegation, DelegationSource, Item, ItemKind, NodeId,
};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_middle::ty::{
    self as ty, AssocKind, DelegationInhFuncKind, Ty, TyCtxt, TypeRelativeDelegationRes,
    TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{ErrorGuaranteed, Span};

use crate::delegation::generics::GenericsGenerationResults;
use crate::delegation::resolution::resolver::DelegationResolver;
use crate::diagnostics::{
    AmbiguousDelegationToInherentImpl, CycleInDelegationSignatureResolution,
    DelegationAttemptedBlockWithDefsDeletion, DelegationAttemptedBlockWithDefsRelowering,
    DelegationBlockSpecifiedWhenNoParams, UnresolvedDelegationCallee,
};

/// Simple (hack or heuristic) resolution of some delegations to inherent impls
/// while correct resolution through `ProbeContext` is not available
/// during AST -> HIR lowering due to query cycles.
/// FIXME(fn_delegation): correct resolution through `ProbeContext` engine
pub(crate) fn resolve_type_relative_delegations(
    tcx: TyCtxt<'_>,
    _: (),
) -> FxIndexMap<LocalDefId, TypeRelativeDelegationRes> {
    let ast_index = tcx.index_ast(());
    let resolutions = tcx.resolutions(());

    let infos = &resolutions.delegation_infos;
    let inh_fns = &resolutions.delegation_inh_functions_map;

    let mut type_relative_resolutions: FxIndexMap<LocalDefId, TypeRelativeDelegationRes> =
        Default::default();

    for (&def_id, res) in infos {
        match res.resolution {
            ty::DelegationResolution::Error(..) | ty::DelegationResolution::Full(..) => continue,
            ty::DelegationResolution::Partial => {
                let Some(r_and_owner) = ast_index.get(def_id).map(Steal::borrow) else {
                    unreachable!("ast index must contain delegations");
                };

                let (r, owner) = &*r_and_owner;

                let delegation = match owner {
                    AstOwner::Item(Item { kind: ItemKind::Delegation(d), .. })
                    | AstOwner::TraitItem(Item { kind: AssocItemKind::Delegation(d), .. })
                    | AstOwner::ImplItem(Item { kind: AssocItemKind::Delegation(d), .. }) => d,
                    _ => unreachable!("we are processing only delegations"),
                };

                let res = r.partial_res_map.get(&delegation.id);
                let res = res.and_then(|res| res.base_res().opt_def_id());
                let ident = delegation.path.segments.last().map(|s| s.ident);

                let span = delegation.last_segment_span();

                let ambig_error_res = || {
                    TypeRelativeDelegationRes::Ambig(
                        tcx.dcx().span_delayed_bug(span, "ambiguous delegation to inherent impl"),
                    )
                };

                let default_error_res =
                    || {
                        TypeRelativeDelegationRes::Error(tcx.dcx().span_delayed_bug(
                            span,
                            "failed to resolve delegation to inherent impl",
                        ))
                    };

                let res = if let Some(res) = res
                    && let Some(ident) = ident
                {
                    match res.as_local() {
                        Some(local_def_id) => {
                            let res = inh_fns.get(&local_def_id).and_then(|map| map.get(&ident));

                            match res {
                                Some(res) => match res {
                                    DelegationInhFuncKind::Ambig => ambig_error_res(),
                                    DelegationInhFuncKind::Single(res) => {
                                        TypeRelativeDelegationRes::Ok(res.to_def_id())
                                    }
                                },
                                _ => default_error_res(),
                            }
                        }
                        None => {
                            let mut sig_res = None;
                            'inh_loop: for inh_impl_id in tcx.inherent_impls(res) {
                                let assoc_items = tcx.associated_items(*inh_impl_id);
                                let mut candidates = assoc_items
                                    .filter_by_name_unhygienic(ident.name)
                                    .filter(|it| matches!(it.kind, AssocKind::Fn { .. }));

                                while let Some(candidate) = candidates.next() {
                                    if sig_res.is_some() {
                                        sig_res = Some(ambig_error_res());
                                        break 'inh_loop;
                                    } else {
                                        sig_res =
                                            Some(TypeRelativeDelegationRes::Ok(candidate.def_id));
                                    }
                                }
                            }

                            sig_res.unwrap_or_else(default_error_res)
                        }
                    }
                } else {
                    default_error_res()
                };

                type_relative_resolutions.insert(def_id, res);
            }
        }
    }

    type_relative_resolutions
}

/// Summary info about function parameters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) struct ParamInfo {
    /// The number of function parameters, including any C variadic `...` parameter.
    pub param_count: usize,

    /// Whether the function arguments end in a C variadic `...` parameter.
    pub c_variadic: bool,

    /// The index of the splatted parameter, if any.
    pub splatted: Option<u8>,
}

#[derive(Default, Debug)]
pub(super) struct SigMapping {
    pub map_return: bool,
    pub arguments_to_map: FxIndexSet<usize>,
}

pub(super) struct DelegationResolution {
    pub sig_id: DefId,
    pub is_method: bool,
    pub param_info: ParamInfo,
    pub span: Span,
    pub call_path_res: DefId,
    pub source: DelegationSource,
    pub parent: LocalDefId,
    pub sig_mapping: SigMapping,
}

pub(super) mod resolver {
    use rustc_ast::NodeId;
    use rustc_hir::def_id::{DefId, LocalDefId};
    use rustc_middle::ty::TyCtxt;
    use rustc_span::ErrorGuaranteed;

    use crate::LoweringContext;

    /// Abstracts operations that are needed for delegation's resolution, so resolution
    /// is independent of `LoweringContext`. Placed in a separate module so `LoweringContext`
    /// can not be accessed directly.
    pub(crate) struct DelegationResolver<'a, 'hir>(&'a LoweringContext<'a, 'hir>);

    impl<'a, 'tcx> DelegationResolver<'a, 'tcx> {
        pub(crate) fn new(ctx: &'a LoweringContext<'a, 'tcx>) -> Self {
            DelegationResolver(ctx)
        }

        #[inline]
        pub(crate) fn tcx(&self) -> TyCtxt<'tcx> {
            self.0.tcx
        }

        #[inline]
        pub(crate) fn owner_id(&self) -> LocalDefId {
            self.0.owner.def_id
        }

        /// (from `tests\ui\delegation\target-expr-removal-defs-inside.rs`):
        /// ```rust
        /// reuse impl Trait for S1 {
        ///     some::path::<{ fn foo() {} }>::xd();
        ///     fn foo() {}
        ///     self.0
        /// }
        /// ```
        ///
        /// Constant from unresolved path will be in `node_id_to_def_id`,
        /// `fn foo() {}` will not be in `node_id_to_def_id` but will be in `owners`,
        /// both have `LocalDefId`, so we check those two maps.
        #[inline]
        pub(crate) fn is_definition(&self, id: NodeId) -> bool {
            self.0.resolver.owners.contains_key(&id)
                || self.0.owner.node_id_to_def_id.contains_key(&id)
        }

        #[inline]
        pub(crate) fn get_resolution_id(&self, id: NodeId) -> Result<DefId, ErrorGuaranteed> {
            self.0.get_partial_res(id).and_then(|r| r.expect_full_res().opt_def_id()).ok_or_else(
                || self.tcx().dcx().delayed_bug(format!("failed to resolve node {id:?}")),
            )
        }

        pub(crate) fn opt_resolution_id(&self, id: NodeId) -> Option<DefId> {
            self.0.get_partial_res(id).and_then(|r| r.full_res()).and_then(|r| r.opt_def_id())
        }
    }
}

impl<'tcx> DelegationResolver<'_, 'tcx> {
    pub(super) fn resolve_delegation(
        &self,
        delegation: &Delegation,
        span: Span,
    ) -> Result<(DelegationResolution, GenericsGenerationResults<'tcx>), ErrorGuaranteed> {
        let tcx = self.tcx();
        let def_id = self.owner_id();

        // Delegation can be missing from the `delegations_resolutions` table
        // in illegal places such as function bodies in extern blocks (see #151356).
        let sig_id = self.resolve_delegation_sig(def_id, span)?;

        let create_invalid_path_error =
            || tcx.dcx().span_delayed_bug(span, "invalid delegation path");

        match &delegation.path.segments[..] {
            [] => return Err(create_invalid_path_error()),
            [child] => {
                let res = self.get_resolution_id(child.id)?;
                if tcx.def_kind(res) != DefKind::Fn {
                    return Err(create_invalid_path_error());
                }
            }
            [.., parent, _] => {
                let child_res = self.get_call_path_res(delegation, span)?;
                let parent_res = self.get_resolution_id(parent.id)?;

                match (tcx.def_kind(child_res), tcx.def_kind(parent_res)) {
                    (DefKind::Fn, DefKind::Mod) => {}
                    (DefKind::AssocFn, DefKind::Trait | DefKind::Struct | DefKind::Enum) => {}
                    _ => return Err(create_invalid_path_error()),
                }
            }
        }

        self.check_for_cycles(sig_id, span)?;

        let is_method = tcx.is_method(sig_id);
        let sig = tcx.fn_sig(sig_id).skip_binder().skip_binder();
        let param_count = sig.inputs().len() + usize::from(sig.c_variadic());
        let parent = tcx.local_parent(def_id);

        let (should_generate_block, contains_defs) =
            self.check_block_soundness(delegation, sig_id, is_method, param_count)?;

        let res = DelegationResolution {
            is_method,
            span,
            sig_id,
            parent,
            // FIXME(splat): use `sig.splatted()` once FnSig has it
            param_info: ParamInfo { param_count, c_variadic: sig.c_variadic(), splatted: None },
            source: delegation.source,
            call_path_res: self.get_call_path_res(delegation, span)?,
            sig_mapping: self.create_sig_mapping(
                delegation,
                span,
                should_generate_block,
                parent,
                sig,
                contains_defs,
            )?,
        };

        Ok((res, self.resolve_and_generate_generics(delegation, sig_id, span)?))
    }

    fn get_call_path_res(
        &self,
        delegation: &Delegation,
        span: Span,
    ) -> Result<DefId, ErrorGuaranteed> {
        self.opt_resolution_id(delegation.id)
            .map(|id| Ok(id))
            .unwrap_or_else(|| self.resolve_delegation_sig(self.owner_id(), span))
    }

    fn resolve_delegation_sig(
        &self,
        def_id: LocalDefId,
        span: Span,
    ) -> Result<DefId, ErrorGuaranteed> {
        match self.tcx().resolutions(()).delegation_infos.get(&def_id) {
            Some(res) => match res.resolution {
                ty::DelegationResolution::Error(err) => Err(err),
                ty::DelegationResolution::Full(def_id) => Ok(def_id),
                ty::DelegationResolution::Partial => {
                    self.resolve_type_relative_delegation_sig(def_id, span)
                }
            },
            None => Err(self.create_unresolved_error(def_id, span)),
        }
    }

    fn create_unresolved_error(&self, def_id: LocalDefId, span: Span) -> ErrorGuaranteed {
        self.tcx().dcx().span_delayed_bug(span, format!("unresolved delegation {def_id:?}"))
    }

    fn resolve_type_relative_delegation_sig(
        &self,
        def_id: LocalDefId,
        span: Span,
    ) -> Result<DefId, ErrorGuaranteed> {
        let tcx = self.tcx();

        if matches!(tcx.def_kind(tcx.local_parent(def_id)), DefKind::Impl { of_trait: true }) {
            return Err(self.create_unresolved_error(def_id, span));
        }

        match tcx.resolve_type_relative_delegations(()).get(&def_id) {
            Some(res) => match *res {
                TypeRelativeDelegationRes::Ok(sig_id) => Ok(sig_id),
                TypeRelativeDelegationRes::Error(err) => Err(err),
                TypeRelativeDelegationRes::Ambig(_) => {
                    Err(tcx.dcx().emit_err(AmbiguousDelegationToInherentImpl { span }))
                }
            },
            None => Err(self.create_unresolved_error(def_id, span)),
        }
    }

    fn check_for_cycles(&self, mut def_id: DefId, span: Span) -> Result<(), ErrorGuaranteed> {
        let tcx = self.tcx();
        let mut visited: FxHashSet<DefId> = Default::default();
        let delegation_infos = &tcx.resolutions(()).delegation_infos;

        loop {
            visited.insert(def_id);

            // If def_id is in local crate and it corresponds to another delegation
            // it means that we refer to another delegation as a callee, so in order to obtain
            // a signature DefId we obtain NodeId of the callee delegation and try to get signature from it.
            if let Some(local_id) = def_id.as_local()
                && delegation_infos.contains_key(&local_id)
                && let Ok(id) = self.resolve_delegation_sig(local_id, span)
            {
                def_id = id;
                if visited.contains(&def_id) {
                    return Err(match visited.len() {
                        1 => tcx.dcx().emit_err(UnresolvedDelegationCallee { span }),
                        _ => tcx.dcx().emit_err(CycleInDelegationSignatureResolution { span }),
                    });
                }
            } else {
                return Ok(());
            }
        }
    }

    fn check_block_soundness(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
        is_method: bool,
        param_count: usize,
    ) -> Result<(/* should generate block */ bool, /* contains defs */ bool), ErrorGuaranteed> {
        let tcx = self.tcx();
        let should_generate_block = is_method
            || matches!(tcx.def_kind(sig_id), DefKind::Fn)
            || matches!(delegation.source, DelegationSource::Single);

        let Some(block) = &delegation.body else { return Ok((should_generate_block, false)) };

        // Report an error if user has explicitly specified delegation's target expression
        // in a single delegation when reused function has no params.
        if param_count == 0 && should_generate_block {
            let err = DelegationBlockSpecifiedWhenNoParams { span: block.span };
            return Err(tcx.dcx().emit_err(err));
        }

        struct DefinitionsFinder<'a, 'hir> {
            resolver: &'a DelegationResolver<'a, 'hir>,
        }

        impl<'a> Visitor<'a> for DefinitionsFinder<'a, '_> {
            type Result = ControlFlow<()>;

            fn visit_id(&mut self, id: NodeId) -> Self::Result {
                match self.resolver.is_definition(id) {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            }
        }

        let mut collector = DefinitionsFinder { resolver: self };

        let contains_defs = collector.visit_block(block).is_break();

        // If there are definitions inside and we can't delete target expression, then report an error.
        // FIXME(fn_delegation): support deletion of target expression with defs inside.
        if should_generate_block || !contains_defs {
            Ok((should_generate_block, contains_defs))
        } else {
            Err(tcx.dcx().emit_err(DelegationAttemptedBlockWithDefsDeletion { span: block.span }))
        }
    }

    fn create_sig_mapping(
        &self,
        delegation: &Delegation,
        span: Span,
        should_generate_block: bool,
        parent: LocalDefId,
        sig: ty::FnSig<'tcx>,
        contains_defs: bool,
    ) -> Result<SigMapping, ErrorGuaranteed> {
        let mut mapping = SigMapping::default();
        if should_generate_block {
            mapping.arguments_to_map.insert(0);
        }

        if self.can_perform_self_mapping(delegation, parent) {
            /// Finds `Self` generic param only in ADT or references, so we avoid cases like
            /// `Self::Item` which will return true if `output.contains(...)` will be used.
            struct SelfFinder;

            impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for SelfFinder {
                type Result = ControlFlow<()>;

                fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                    match t.kind() {
                        ty::Adt(_, args) => {
                            if args
                                .iter()
                                .flat_map(|arg| arg.as_type())
                                .any(|type_arg| type_arg.is_self_param())
                            {
                                return ControlFlow::Break(());
                            }

                            t.super_visit_with(self)
                        }
                        ty::Ref(_, ref_t, _) => {
                            if ref_t.is_self_param() {
                                return ControlFlow::Break(());
                            }

                            t.super_visit_with(self)
                        }
                        _ => ControlFlow::Continue(()),
                    }
                }
            }

            impl SelfFinder {
                fn contains_self(t: Ty<'_>) -> bool {
                    t.is_self_param() || t.visit_with(&mut SelfFinder).is_break()
                }
            }

            mapping.map_return = SelfFinder::contains_self(sig.output());

            let arguments_to_map = sig
                .inputs()
                .iter()
                .enumerate()
                .skip(1) // Already checked above.
                .filter_map(|(idx, &param)| SelfFinder::contains_self(param).then_some(idx));

            mapping.arguments_to_map.extend(arguments_to_map);
        }

        // We can't yet map more than one argument if there are definitions inside.
        // FIXME(fn_delegation): support relowering with defs inside
        if contains_defs && mapping.arguments_to_map.len() > 1 {
            let err = DelegationAttemptedBlockWithDefsRelowering { span };
            let err = self.tcx().dcx().emit_err(err);
            return Err(err);
        }

        Ok(mapping)
    }

    fn can_perform_self_mapping(&self, delegation: &Delegation, parent: LocalDefId) -> bool {
        // Heuristic: don't do wrapping if there is no target expression.
        if delegation.body.is_none() {
            return false;
        }

        let tcx = self.tcx();

        // Apply wrapping for delegations inside
        // 1) Trait impls, as the return type of both signature function
        //    and generated delegation has `Self` generic param returned
        //    (checked below).
        //    FIXME(fn_delegation): think of enabling wrapping in more scenarios:
        //      trait-(impl)-to-free
        //      trait-(impl)-to-inherent
        //      inherent-to-free
        // 2) Inherent methods when delegating to trait, as we change the type of
        //    `Self` to type of struct or enum we delegate from.
        if !matches!(tcx.def_kind(parent), DefKind::Impl { .. }) {
            return false;
        }

        // Check that delegation path resolves to a trait AssocFn, not to a free method.
        // After previous check we are sure that `sig_id` and `delegation.id`
        // point to the same function.
        self.opt_resolution_id(delegation.id)
            .map(|id| {
                tcx.def_kind(id) == DefKind::AssocFn
                    && tcx.def_kind(tcx.parent(id)) == DefKind::Trait
            })
            .unwrap_or(false)
    }
}
