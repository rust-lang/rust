use std::ops::ControlFlow;

use ast::visit::Visitor;
use hir::def::DefKind;
use rustc_ast as ast;
use rustc_ast::node_id::NodeMap;
use rustc_ast::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::span_bug;
use rustc_middle::ty::PerOwnerResolverData;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{ErrorGuaranteed, Span};

use crate::LoweringContext;
use crate::diagnostics::{
    CycleInDelegationSignatureResolution, DelegationAttemptedBlockWithDefsDeletion,
    DelegationBlockSpecifiedWhenNoParams, UnresolvedDelegationCallee,
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

impl<'hir> LoweringContext<'_, 'hir> {
    pub(super) fn is_method(&self, def_id: DefId, span: Span) -> bool {
        match self.tcx.def_kind(def_id) {
            DefKind::Fn => false,
            DefKind::AssocFn => self.tcx.associated_item(def_id).is_method(),
            _ => span_bug!(span, "unexpected DefKind for delegation item"),
        }
    }

    pub(super) fn check_for_cycles(
        &self,
        mut def_id: DefId,
        span: Span,
    ) -> Result<(), ErrorGuaranteed> {
        let mut visited: FxHashSet<DefId> = Default::default();

        loop {
            visited.insert(def_id);

            // If def_id is in local crate and it corresponds to another delegation
            // it means that we refer to another delegation as a callee, so in order to obtain
            // a signature DefId we obtain NodeId of the callee delegation and try to get signature from it.
            if let Some(local_id) = def_id.as_local()
                && let Some(info) = self.tcx.resolutions(()).delegation_infos.get(&local_id)
                && let Ok(id) = info.resolution_id
            {
                def_id = id;
                if visited.contains(&def_id) {
                    return Err(match visited.len() {
                        1 => self.dcx().emit_err(UnresolvedDelegationCallee { span }),
                        _ => self.dcx().emit_err(CycleInDelegationSignatureResolution { span }),
                    });
                }
            } else {
                return Ok(());
            }
        }
    }

    pub(super) fn check_block_soundness(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
        is_method: bool,
        param_count: usize,
    ) -> bool {
        let Some(block) = delegation.body.as_ref() else { return true };
        let should_generate_block = self.should_generate_block(delegation, sig_id, is_method);

        // Report an error if user has explicitly specified delegation's target expression
        // in a single delegation when reused function has no params.
        if param_count == 0 && should_generate_block {
            self.dcx().emit_err(DelegationBlockSpecifiedWhenNoParams { span: block.span });
            return false;
        }

        struct DefinitionsFinder<'a> {
            all_owners: &'a NodeMap<PerOwnerResolverData<'a>>,
            // `self.owner.node_id_to_def_id`
            nested_def_ids: &'a NodeMap<LocalDefId>,
        }

        impl<'a> ast::visit::Visitor<'a> for DefinitionsFinder<'a> {
            type Result = ControlFlow<()>;

            fn visit_id(&mut self, id: NodeId) -> Self::Result {
                /*
                    (from `tests\ui\delegation\target-expr-removal-defs-inside.rs`):
                    ```rust
                        reuse impl Trait for S1 {
                            some::path::<{ fn foo() {} }>::xd();
                            fn foo() {}
                            self.0
                        }
                    ```

                    Constant from unresolved path will be in `nested_owners`,
                    `fn foo() {}` will not be in `nested_owners` but will be in `owners`,
                    both have `LocalDefId`, so we check those two maps.
                */
                match self.all_owners.contains_key(&id) || self.nested_def_ids.contains_key(&id) {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            }
        }

        let mut collector = DefinitionsFinder {
            all_owners: &self.resolver.owners,
            nested_def_ids: &self.owner.node_id_to_def_id,
        };

        let contains_defs = collector.visit_block(block).is_break();

        // If there are definitions inside and we can't delete target expression, so report an error.
        // FIXME(fn_delegation): support deletion of target expression with defs inside.
        if !should_generate_block && contains_defs {
            self.dcx().emit_err(DelegationAttemptedBlockWithDefsDeletion { span: block.span });
            return false;
        }

        true
    }

    pub(super) fn should_generate_block(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
        is_method: bool,
    ) -> bool {
        is_method
            || matches!(self.tcx.def_kind(sig_id), DefKind::Fn)
            || matches!(delegation.source, DelegationSource::Single)
    }

    pub(super) fn get_resolution_id(&self, node_id: NodeId) -> Option<DefId> {
        self.get_partial_res(node_id).and_then(|r| r.expect_full_res().opt_def_id())
    }

    /// Returns function parameter info, including C variadic `...` and `#[splat]` if present.
    pub(super) fn param_info(&self, def_id: DefId) -> ParamInfo {
        let sig = self.tcx.fn_sig(def_id).skip_binder().skip_binder();

        ParamInfo {
            param_count: sig.inputs().len() + usize::from(sig.c_variadic()),
            c_variadic: sig.c_variadic(),
            splatted: sig.splatted(),
        }
    }

    pub(super) fn should_wrap_return_value(
        &self,
        delegation: &Delegation,
    ) -> Option<(LocalDefId, bool)> {
        // Heuristic: don't do wrapping if there is no target expression.
        if delegation.body.is_none() {
            return None;
        }

        let tcx = self.tcx;
        let parent = tcx.local_parent(self.owner.def_id);
        let parent_kind = tcx.def_kind(parent);

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
            return None;
        }

        let is_trait_impl = parent_kind == DefKind::Impl { of_trait: true };

        // Check that delegation path resolves to a trait AssocFn, not to a free method.
        Some((parent, is_trait_impl)).filter(|_| {
            self.get_resolution_id(delegation.id).is_some_and(|id| {
                tcx.def_kind(id) == DefKind::AssocFn
                    // Check that the return type of the callee is `Self` param.
                    // After previous check we are sure that `sig_id` and `delegation.id`
                    // point to the same function.
                    && tcx.def_kind(tcx.parent(id)) == DefKind::Trait
                    && tcx.fn_sig(id).skip_binder().output().skip_binder().is_param(0)
            })
        })
    }
}
