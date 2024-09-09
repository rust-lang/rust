//! Diagnostics emitted during DefMap construction.

use std::ops::Not;

use cfg::{CfgExpr, CfgOptions};
use hir_expand::{attrs::AttrId, ExpandErrorKind, MacroCallKind};
use la_arena::Idx;
use syntax::ast;

use crate::{
    item_tree::{self, AttrOwner, ItemTreeId, TreeId},
    nameres::LocalModuleId,
    path::ModPath,
    AstId,
};

#[derive(Debug, PartialEq, Eq)]
pub enum DefDiagnosticKind {
    UnresolvedModule { ast: AstId<ast::Module>, candidates: Box<[String]> },
    UnresolvedExternCrate { ast: AstId<ast::ExternCrate> },
    UnresolvedImport { id: ItemTreeId<item_tree::Use>, index: Idx<ast::UseTree> },
    UnconfiguredCode { tree: TreeId, item: AttrOwner, cfg: CfgExpr, opts: CfgOptions },
    UnresolvedMacroCall { ast: MacroCallKind, path: ModPath },
    UnimplementedBuiltinMacro { ast: AstId<ast::Macro> },
    InvalidDeriveTarget { ast: AstId<ast::Item>, id: usize },
    MalformedDerive { ast: AstId<ast::Adt>, id: usize },
    MacroDefError { ast: AstId<ast::Macro>, message: String },
    MacroError { ast: AstId<ast::Item>, path: ModPath, err: ExpandErrorKind },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefDiagnostics(Option<triomphe::Arc<Box<[DefDiagnostic]>>>);

impl DefDiagnostics {
    pub fn new(diagnostics: Vec<DefDiagnostic>) -> Self {
        Self(
            diagnostics
                .is_empty()
                .not()
                .then(|| triomphe::Arc::new(diagnostics.into_boxed_slice())),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = &DefDiagnostic> {
        self.0.as_ref().into_iter().flat_map(|it| &***it)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DefDiagnostic {
    pub in_module: LocalModuleId,
    pub kind: DefDiagnosticKind,
}

impl DefDiagnostic {
    pub(super) fn unresolved_module(
        container: LocalModuleId,
        declaration: AstId<ast::Module>,
        candidates: Box<[String]>,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::UnresolvedModule { ast: declaration, candidates },
        }
    }

    pub(super) fn unresolved_extern_crate(
        container: LocalModuleId,
        declaration: AstId<ast::ExternCrate>,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::UnresolvedExternCrate { ast: declaration },
        }
    }

    pub(super) fn unresolved_import(
        container: LocalModuleId,
        id: ItemTreeId<item_tree::Use>,
        index: Idx<ast::UseTree>,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnresolvedImport { id, index } }
    }

    pub fn macro_error(
        container: LocalModuleId,
        ast: AstId<ast::Item>,
        path: ModPath,
        err: ExpandErrorKind,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::MacroError { ast, path, err } }
    }

    pub fn unconfigured_code(
        container: LocalModuleId,
        tree: TreeId,
        item: AttrOwner,
        cfg: CfgExpr,
        opts: CfgOptions,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::UnconfiguredCode { tree, item, cfg, opts },
        }
    }

    // FIXME: Whats the difference between this and unresolved_proc_macro
    pub(crate) fn unresolved_macro_call(
        container: LocalModuleId,
        ast: MacroCallKind,
        path: ModPath,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnresolvedMacroCall { ast, path } }
    }

    pub(super) fn unimplemented_builtin_macro(
        container: LocalModuleId,
        ast: AstId<ast::Macro>,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnimplementedBuiltinMacro { ast } }
    }

    pub(super) fn invalid_derive_target(
        container: LocalModuleId,
        ast: AstId<ast::Item>,
        id: AttrId,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::InvalidDeriveTarget { ast, id: id.ast_index() },
        }
    }

    pub(super) fn malformed_derive(
        container: LocalModuleId,
        ast: AstId<ast::Adt>,
        id: AttrId,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::MalformedDerive { ast, id: id.ast_index() },
        }
    }
}
