use std::ops::ControlFlow;

use ast::visit::Visitor;
use hir::def::DefKind;
use rustc_ast::{self as ast, Delegation, DelegationSource, NodeId};
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_hir as hir;
use rustc_middle::ty::Ty;
use rustc_middle::{span_bug, ty};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{ErrorGuaranteed, Span, kw};

use crate::delegation::generics::GenericsGenerationResults;
use crate::delegation::resolution::resolver::DelegationResolver;
use crate::diagnostics::{
    CycleInDelegationSignatureResolution, DelegationAttemptedBlockWithDefsDeletion,
    DelegationAttemptedBlockWithDefsRelowering, DelegationBlockSpecifiedWhenNoParams,
    UnresolvedDelegationCallee,
};

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

#[derive(Default)]
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
        let sig_id = tcx
            .resolutions(())
            .delegation_infos
            .get(&def_id)
            .map(|info| {
                info.resolution_id.and_then(|id| self.check_for_cycles(id, span).map(|_| id))
            })
            .unwrap_or_else(|| {
                Err(tcx.dcx().span_delayed_bug(
                    span,
                    format!("delegation resolution record was not found for {:?}", def_id),
                ))
            })?;

        let is_method = match tcx.def_kind(sig_id) {
            DefKind::Fn => false,
            DefKind::AssocFn => tcx.associated_item(sig_id).is_method(),
            _ => span_bug!(span, "unexpected DefKind for delegation item"),
        };

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
            call_path_res: self.get_resolution_id(delegation.id)?,
            sig_mapping: self.create_sig_mapping(
                delegation,
                span,
                should_generate_block,
                parent,
                sig,
                contains_defs,
            )?,
        };

        Ok((res, self.resolve_and_generate_generics(delegation, sig_id)?))
    }

    fn check_for_cycles(&self, mut def_id: DefId, span: Span) -> Result<(), ErrorGuaranteed> {
        let tcx = self.tcx();
        let mut visited: FxHashSet<DefId> = Default::default();

        loop {
            visited.insert(def_id);

            // If def_id is in local crate and it corresponds to another delegation
            // it means that we refer to another delegation as a callee, so in order to obtain
            // a signature DefId we obtain NodeId of the callee delegation and try to get signature from it.
            if let Some(local_id) = def_id.as_local()
                && let Some(info) = tcx.resolutions(()).delegation_infos.get(&local_id)
                && let Ok(id) = info.resolution_id
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

        if self.can_perform_self_mapping(delegation, parent)? {
            // FIXME(fn_delegation): support heuristics for mapping of complex
            // return types: `Self` -> `Box<Arc<Rc<Self>>>`
            mapping.map_return = sig.output().is_param(0);

            let self_param = Ty::new_param(self.tcx(), 0, kw::SelfUpper);
            let arguments_to_map = sig
                .inputs()
                .iter()
                .enumerate()
                .skip(1) // Already checked above.
                .filter_map(|(idx, param)| param.contains(self_param).then_some(idx));

            mapping.arguments_to_map.extend(arguments_to_map);
        }

        // We can't yet map more than one argument if there are definitions inside.
        // FIXME(fn_delegation): support relowering with defs inside
        if contains_defs && mapping.arguments_to_map.len() > 1 {
            return Err(self
                .tcx()
                .dcx()
                .emit_err(DelegationAttemptedBlockWithDefsRelowering { span }));
        }

        Ok(mapping)
    }

    fn can_perform_self_mapping(
        &self,
        delegation: &Delegation,
        parent: LocalDefId,
    ) -> Result<bool, ErrorGuaranteed> {
        // Heuristic: don't do wrapping if there is no target expression.
        if delegation.body.is_none() {
            return Ok(false);
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
            return Ok(false);
        }

        // Check that delegation path resolves to a trait AssocFn, not to a free method.
        // After previous check we are sure that `sig_id` and `delegation.id`
        // point to the same function.
        let id = self.get_resolution_id(delegation.id)?;
        Ok(tcx.def_kind(id) == DefKind::AssocFn && tcx.def_kind(tcx.parent(id)) == DefKind::Trait)
    }
}
