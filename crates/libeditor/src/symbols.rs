use smol_str::SmolStr;
use libsyntax2::{
    SyntaxKind, SyntaxNodeRef, SyntaxRoot, AstNode,
    ast::{self, NameOwner},
    algo::{
        visit::{visitor, Visitor},
        walk::{walk, WalkEvent},
    },
};
use TextRange;

#[derive(Debug)]
pub struct FileSymbol {
    pub parent: Option<usize>,
    pub name: SmolStr,
    pub name_range: TextRange,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}


pub fn file_symbols(file: &ast::File) -> Vec<FileSymbol> {
    let mut res = Vec::new();
    let mut stack = Vec::new();
    let syntax = file.syntax();

    for event in walk(syntax.as_ref()) {
        match event {
            WalkEvent::Enter(node) => {
                match to_symbol(node) {
                    Some(mut symbol) => {
                        symbol.parent = stack.last().map(|&n| n);
                        stack.push(res.len());
                        res.push(symbol);
                    }
                    None => (),
                }
            }
            WalkEvent::Exit(node) => {
                if to_symbol(node).is_some() {
                    stack.pop().unwrap();
                }
            }
        }
    }
    res
}

fn to_symbol(node: SyntaxNodeRef) -> Option<FileSymbol> {
    fn decl<'a, N: NameOwner<&'a SyntaxRoot>>(node: N) -> Option<FileSymbol> {
        let name = node.name()?;
        Some(FileSymbol {
            parent: None,
            name: name.text(),
            name_range: name.syntax().range(),
            node_range: node.syntax().range(),
            kind: node.syntax().kind(),
        })
    }

    visitor()
        .visit(decl::<ast::FnDef<_>>)
        .visit(decl::<ast::StructDef<_>>)
        .visit(decl::<ast::EnumDef<_>>)
        .visit(decl::<ast::TraitDef<_>>)
        .visit(decl::<ast::Module<_>>)
        .visit(decl::<ast::TypeDef<_>>)
        .visit(decl::<ast::ConstDef<_>>)
        .visit(decl::<ast::StaticDef<_>>)
        .accept(node)?
}
