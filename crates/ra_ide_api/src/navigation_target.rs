use ra_db::FileId;
use ra_syntax::{
    SyntaxNode, AstNode, SmolStr, TextRange, ast,
    SyntaxKind::{self, NAME},
};
use hir::{ModuleSource};

use crate::{FileSymbol, db::RootDatabase};

/// `NavigationTarget` represents and element in the editor's UI which you can
/// click on to navigate to a particular piece of code.
///
/// Typically, a `NavigationTarget` corresponds to some element in the source
/// code, like a function or a struct, but this is not strictly required.
#[derive(Debug, Clone)]
pub struct NavigationTarget {
    file_id: FileId,
    name: SmolStr,
    kind: SyntaxKind,
    full_range: TextRange,
    focus_range: Option<TextRange>,
}

impl NavigationTarget {
    pub fn name(&self) -> &SmolStr {
        &self.name
    }

    pub fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    pub fn full_range(&self) -> TextRange {
        self.full_range
    }

    /// A "most interesting" range withing the `range_full`.
    ///
    /// Typically, `range` is the whole syntax node, including doc comments, and
    /// `focus_range` is the range of the identifier.
    pub fn focus_range(&self) -> Option<TextRange> {
        self.focus_range
    }

    pub(crate) fn from_symbol(symbol: FileSymbol) -> NavigationTarget {
        NavigationTarget {
            file_id: symbol.file_id,
            name: symbol.name.clone(),
            kind: symbol.ptr.kind(),
            full_range: symbol.ptr.range(),
            focus_range: None,
        }
    }

    pub(crate) fn from_scope_entry(
        file_id: FileId,
        entry: &hir::ScopeEntryWithSyntax,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name: entry.name().to_string().into(),
            full_range: entry.ptr().range(),
            focus_range: None,
            kind: NAME,
        }
    }

    pub(crate) fn from_module(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let (file_id, source) = module.definition_source(db);
        let name = module
            .name(db)
            .map(|it| it.to_string().into())
            .unwrap_or_default();
        match source {
            ModuleSource::SourceFile(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
            ModuleSource::Module(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
        }
    }

    pub(crate) fn from_module_to_decl(db: &RootDatabase, module: hir::Module) -> NavigationTarget {
        let name = module
            .name(db)
            .map(|it| it.to_string().into())
            .unwrap_or_default();
        if let Some((file_id, source)) = module.declaration_source(db) {
            return NavigationTarget::from_syntax(file_id, name, None, source.syntax());
        }
        NavigationTarget::from_module(db, module)
    }

    pub(crate) fn from_function(db: &RootDatabase, func: hir::Function) -> NavigationTarget {
        let (file_id, fn_def) = func.source(db);
        NavigationTarget::from_named(file_id.original_file(db), &*fn_def)
    }

    // TODO once Def::Item is gone, this should be able to always return a NavigationTarget
    pub(crate) fn from_def(
        db: &RootDatabase,
        module_def: hir::ModuleDef,
    ) -> Option<NavigationTarget> {
        match module_def {
            hir::ModuleDef::Module(module) => Some(NavigationTarget::from_module(db, module)),
            hir::ModuleDef::Function(func) => Some(NavigationTarget::from_function(db, func)),
            hir::ModuleDef::Struct(s) => {
                let (file_id, node) = s.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::Const(s) => {
                let (file_id, node) = s.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::Static(s) => {
                let (file_id, node) = s.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::Enum(e) => {
                let (file_id, node) = e.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::EnumVariant(var) => {
                let (file_id, node) = var.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::Trait(e) => {
                let (file_id, node) = e.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
            hir::ModuleDef::Type(e) => {
                let (file_id, node) = e.source(db);
                Some(NavigationTarget::from_named(
                    file_id.original_file(db),
                    &*node,
                ))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn assert_match(&self, expected: &str) {
        let actual = self.debug_render();
        test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
    }

    #[cfg(test)]
    pub(crate) fn debug_render(&self) -> String {
        let mut buf = format!(
            "{} {:?} {:?} {:?}",
            self.name(),
            self.kind(),
            self.file_id(),
            self.full_range()
        );
        if let Some(focus_range) = self.focus_range() {
            buf.push_str(&format!(" {:?}", focus_range))
        }
        buf
    }

    fn from_named(file_id: FileId, node: &impl ast::NameOwner) -> NavigationTarget {
        let name = node.name().map(|it| it.text().clone()).unwrap_or_default();
        let focus_range = node.name().map(|it| it.syntax().range());
        NavigationTarget::from_syntax(file_id, name, focus_range, node.syntax())
    }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        node: &SyntaxNode,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind: node.kind(),
            full_range: node.range(),
            focus_range,
            // ptr: Some(LocalSyntaxPtr::new(node)),
        }
    }
}
