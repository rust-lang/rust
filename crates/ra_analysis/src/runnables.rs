use ra_syntax::{
    ast::{self, AstNode, NameOwner, ModuleItemOwner},
    SourceFileNode, TextRange, SyntaxNodeRef,
    TextUnit,
};
use crate::{
    Analysis, FileId, FilePosition
};

#[derive(Debug)]
pub struct Runnable {
    pub range: TextRange,
    pub kind: RunnableKind,
}

#[derive(Debug)]
pub enum RunnableKind {
    Test { name: String },
    TestMod { path: String },
    Bin,
}

pub fn runnables(
    analysis: &Analysis,
    file_node: &SourceFileNode,
    file_id: FileId,
) -> Vec<Runnable> {
    file_node
        .syntax()
        .descendants()
        .filter_map(|i| runnable(analysis, i, file_id))
        .collect()
}

fn runnable<'a>(analysis: &Analysis, item: SyntaxNodeRef<'a>, file_id: FileId) -> Option<Runnable> {
    if let Some(f) = ast::FnDef::cast(item) {
        let name = f.name()?.text();
        let kind = if name == "main" {
            RunnableKind::Bin
        } else if f.has_atom_attr("test") {
            RunnableKind::Test {
                name: name.to_string(),
            }
        } else {
            return None;
        };
        Some(Runnable {
            range: f.syntax().range(),
            kind,
        })
    } else if let Some(m) = ast::Module::cast(item) {
        if m.item_list()?
            .items()
            .map(ast::ModuleItem::syntax)
            .filter_map(ast::FnDef::cast)
            .any(|f| f.has_atom_attr("test"))
        {
            let postition = FilePosition {
                file_id,
                offset: m.syntax().range().start() + TextUnit::from_usize(1),
            };
            analysis.module_path(postition).ok()?.map(|path| Runnable {
                range: m.syntax().range(),
                kind: RunnableKind::TestMod { path },
            })
        } else {
            None
        }
    } else {
        None
    }
}
