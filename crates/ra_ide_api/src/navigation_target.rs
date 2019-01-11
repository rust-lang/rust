use ra_db::{FileId, LocalSyntaxPtr, Cancelable};
use ra_syntax::{
    SyntaxNode, AstNode, SmolStr, TextRange, ast,
    SyntaxKind::{self, NAME},
};
use hir::{Def, ModuleSource};

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
    range: TextRange,
    focus_range: Option<TextRange>,
    // Should be DefId ideally
    ptr: Option<LocalSyntaxPtr>,
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

    pub fn range(&self) -> TextRange {
        self.range
    }

    /// A "most interesting" range withing the `range`.
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
            range: symbol.ptr.range(),
            focus_range: None,
            ptr: Some(symbol.ptr.clone()),
        }
    }

    pub(crate) fn from_scope_entry(
        file_id: FileId,
        entry: &hir::ScopeEntryWithSyntax,
    ) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name: entry.name().to_string().into(),
            range: entry.ptr().range(),
            focus_range: None,
            kind: NAME,
            ptr: None,
        }
    }

    pub(crate) fn from_module(
        db: &RootDatabase,
        module: hir::Module,
    ) -> Cancelable<NavigationTarget> {
        let (file_id, source) = module.definition_source(db)?;
        let name = module
            .name(db)?
            .map(|it| it.to_string().into())
            .unwrap_or_default();
        let res = match source {
            ModuleSource::SourceFile(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
            ModuleSource::Module(node) => {
                NavigationTarget::from_syntax(file_id, name, None, node.syntax())
            }
        };
        Ok(res)
    }

    // TODO once Def::Item is gone, this should be able to always return a NavigationTarget
    pub(crate) fn from_def(db: &RootDatabase, def: Def) -> Cancelable<Option<NavigationTarget>> {
        let res = match def {
            Def::Struct(s) => {
                let (file_id, node) = s.source(db)?;
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            Def::Enum(e) => {
                let (file_id, node) = e.source(db)?;
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            Def::EnumVariant(ev) => {
                let (file_id, node) = ev.source(db)?;
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            Def::Function(f) => {
                let (file_id, node) = f.source(db)?;
                NavigationTarget::from_named(file_id.original_file(db), &*node)
            }
            Def::Module(m) => NavigationTarget::from_module(db, m)?,
            Def::Item => return Ok(None),
        };
        Ok(Some(res))
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
            range: node.range(),
            focus_range,
            ptr: Some(LocalSyntaxPtr::new(node)),
        }
    }
}
