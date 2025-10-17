/// This module provides types and traits for buffering lints until later in compilation.
use rustc_ast::node_id::NodeId;
use rustc_data_structures::fx::FxIndexMap;
use rustc_error_messages::MultiSpan;
use rustc_lint_defs::{BuiltinLintDiag, Lint, LintId};

use crate::{DynSend, LintDiagnostic, LintDiagnosticBox};

/// We can't implement `LintDiagnostic` for `BuiltinLintDiag`, because decorating some of its
/// variants requires types we don't have yet. So, handle that case separately.
pub enum DecorateDiagCompat {
    Dynamic(Box<dyn for<'a> LintDiagnosticBox<'a, ()> + DynSend + 'static>),
    Builtin(BuiltinLintDiag),
}

impl std::fmt::Debug for DecorateDiagCompat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecorateDiagCompat").finish()
    }
}

impl !LintDiagnostic<'_, ()> for BuiltinLintDiag {}

impl<D: for<'a> LintDiagnostic<'a, ()> + DynSend + 'static> From<D> for DecorateDiagCompat {
    #[inline]
    fn from(d: D) -> Self {
        Self::Dynamic(Box::new(d))
    }
}

impl From<BuiltinLintDiag> for DecorateDiagCompat {
    #[inline]
    fn from(b: BuiltinLintDiag) -> Self {
        Self::Builtin(b)
    }
}

/// Lints that are buffered up early on in the `Session` before the
/// `LintLevels` is calculated.
#[derive(Debug)]
pub struct BufferedEarlyLint {
    /// The span of code that we are linting on.
    pub span: Option<MultiSpan>,

    /// The `NodeId` of the AST node that generated the lint.
    pub node_id: NodeId,

    /// A lint Id that can be passed to
    /// `rustc_lint::early::EarlyContextAndPass::check_id`.
    pub lint_id: LintId,

    /// Customization of the `Diag<'_>` for the lint.
    pub diagnostic: DecorateDiagCompat,
}

#[derive(Default, Debug)]
pub struct LintBuffer {
    pub map: FxIndexMap<NodeId, Vec<BufferedEarlyLint>>,
}

impl LintBuffer {
    pub fn add_early_lint(&mut self, early_lint: BufferedEarlyLint) {
        self.map.entry(early_lint.node_id).or_default().push(early_lint);
    }

    pub fn take(&mut self, id: NodeId) -> Vec<BufferedEarlyLint> {
        // FIXME(#120456) - is `swap_remove` correct?
        self.map.swap_remove(&id).unwrap_or_default()
    }

    pub fn buffer_lint(
        &mut self,
        lint: &'static Lint,
        node_id: NodeId,
        span: impl Into<MultiSpan>,
        decorate: impl Into<DecorateDiagCompat>,
    ) {
        self.add_early_lint(BufferedEarlyLint {
            lint_id: LintId::of(lint),
            node_id,
            span: Some(span.into()),
            diagnostic: decorate.into(),
        });
    }
}
