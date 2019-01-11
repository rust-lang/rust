use ra_db::{FileId, LocalSyntaxPtr, Cancelable};
use ra_syntax::{
    SyntaxNode, AstNode, SmolStr,
    ast
};
use hir::{Def, ModuleSource};

use crate::{
    NavigationTarget,
    FileSymbol,
    db::RootDatabase,
};

impl NavigationTarget {
    pub(crate) fn from_symbol(symbol: FileSymbol) -> NavigationTarget {
        NavigationTarget {
            file_id: symbol.file_id,
            name: symbol.name.clone(),
            kind: symbol.ptr.kind(),
            range: symbol.ptr.range(),
            ptr: Some(symbol.ptr.clone()),
        }
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
            Def::Module(m) => {
                let (file_id, source) = m.definition_source(db)?;
                let name = m
                    .name(db)?
                    .map(|it| it.to_string().into())
                    .unwrap_or_else(|| SmolStr::new(""));
                match source {
                    ModuleSource::SourceFile(node) => {
                        NavigationTarget::from_syntax(file_id, name, node.syntax())
                    }
                    ModuleSource::Module(node) => {
                        NavigationTarget::from_syntax(file_id, name, node.syntax())
                    }
                }
            }
            Def::Item => return Ok(None),
        };
        Ok(Some(res))
    }

    fn from_named(file_id: FileId, node: &impl ast::NameOwner) -> NavigationTarget {
        let name = node
            .name()
            .map(|it| it.text().clone())
            .unwrap_or_else(|| SmolStr::new(""));
        NavigationTarget::from_syntax(file_id, name, node.syntax())
    }

    fn from_syntax(file_id: FileId, name: SmolStr, node: &SyntaxNode) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name,
            kind: node.kind(),
            range: node.range(),
            ptr: Some(LocalSyntaxPtr::new(node)),
        }
    }
}
