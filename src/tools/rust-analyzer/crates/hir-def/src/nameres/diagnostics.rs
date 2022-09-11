//! Diagnostics emitted during DefMap construction.

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use hir_expand::MacroCallKind;
use la_arena::Idx;
use syntax::ast;

use crate::{
    attr::AttrId,
    item_tree::{self, ItemTreeId},
    nameres::LocalModuleId,
    path::ModPath,
    AstId,
};

#[derive(Debug, PartialEq, Eq)]
pub enum DefDiagnosticKind {
    UnresolvedModule { ast: AstId<ast::Module>, candidates: Box<[String]> },

    UnresolvedExternCrate { ast: AstId<ast::ExternCrate> },

    UnresolvedImport { id: ItemTreeId<item_tree::Import>, index: Idx<ast::UseTree> },

    UnconfiguredCode { ast: AstId<ast::Item>, cfg: CfgExpr, opts: CfgOptions },

    UnresolvedProcMacro { ast: MacroCallKind, krate: CrateId },

    UnresolvedMacroCall { ast: MacroCallKind, path: ModPath },

    MacroError { ast: MacroCallKind, message: String },

    UnimplementedBuiltinMacro { ast: AstId<ast::Macro> },

    InvalidDeriveTarget { ast: AstId<ast::Item>, id: u32 },

    MalformedDerive { ast: AstId<ast::Adt>, id: u32 },
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
        id: ItemTreeId<item_tree::Import>,
        index: Idx<ast::UseTree>,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnresolvedImport { id, index } }
    }

    pub fn unconfigured_code(
        container: LocalModuleId,
        ast: AstId<ast::Item>,
        cfg: CfgExpr,
        opts: CfgOptions,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnconfiguredCode { ast, cfg, opts } }
    }

    pub(super) fn unresolved_proc_macro(
        container: LocalModuleId,
        ast: MacroCallKind,
        krate: CrateId,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::UnresolvedProcMacro { ast, krate } }
    }

    pub(super) fn macro_error(
        container: LocalModuleId,
        ast: MacroCallKind,
        message: String,
    ) -> Self {
        Self { in_module: container, kind: DefDiagnosticKind::MacroError { ast, message } }
    }

    pub(super) fn unresolved_macro_call(
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
            kind: DefDiagnosticKind::InvalidDeriveTarget { ast, id: id.ast_index },
        }
    }

    pub(super) fn malformed_derive(
        container: LocalModuleId,
        ast: AstId<ast::Adt>,
        id: AttrId,
    ) -> Self {
        Self {
            in_module: container,
            kind: DefDiagnosticKind::MalformedDerive { ast, id: id.ast_index },
        }
    }
}
