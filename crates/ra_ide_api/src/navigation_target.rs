use ra_db::{FileId, LocalSyntaxPtr, Cancelable};
use ra_syntax::{SyntaxNode, AstNode};
use hir::{Name, Def, ModuleSource};

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
        Ok(match def {
            Def::Struct(s) => {
                let (file_id, node) = s.source(db)?;
                Some(NavigationTarget::from_syntax(
                    s.name(db)?,
                    file_id.original_file(db),
                    node.syntax(),
                ))
            }
            Def::Enum(e) => {
                let (file_id, node) = e.source(db)?;
                Some(NavigationTarget::from_syntax(
                    e.name(db)?,
                    file_id.original_file(db),
                    node.syntax(),
                ))
            }
            Def::EnumVariant(ev) => {
                let (file_id, node) = ev.source(db)?;
                Some(NavigationTarget::from_syntax(
                    ev.name(db)?,
                    file_id.original_file(db),
                    node.syntax(),
                ))
            }
            Def::Function(f) => {
                let (file_id, node) = f.source(db)?;
                let name = f.signature(db).name().clone();
                Some(NavigationTarget::from_syntax(
                    Some(name),
                    file_id.original_file(db),
                    node.syntax(),
                ))
            }
            Def::Module(m) => {
                let (file_id, source) = m.definition_source(db)?;
                let name = m.name(db)?;
                match source {
                    ModuleSource::SourceFile(node) => {
                        Some(NavigationTarget::from_syntax(name, file_id, node.syntax()))
                    }
                    ModuleSource::Module(node) => {
                        Some(NavigationTarget::from_syntax(name, file_id, node.syntax()))
                    }
                }
            }
            Def::Item => None,
        })
    }

    fn from_syntax(name: Option<Name>, file_id: FileId, node: &SyntaxNode) -> NavigationTarget {
        NavigationTarget {
            file_id,
            name: name.map(|n| n.to_string().into()).unwrap_or("".into()),
            kind: node.kind(),
            range: node.range(),
            ptr: Some(LocalSyntaxPtr::new(node)),
        }
    }
}
