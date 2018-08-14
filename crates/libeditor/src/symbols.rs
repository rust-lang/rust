use smol_str::SmolStr;
use libsyntax2::{
    SyntaxKind, SyntaxNodeRef, SyntaxRoot, AstNode,
    ast::{self, NameOwner},
    algo::{
        visit::{visitor, Visitor},
        walk::{walk, WalkEvent, preorder},
    },
    SyntaxKind::*,
};
use TextRange;

#[derive(Debug)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}

#[derive(Debug)]
pub struct FileSymbol {
    pub name: SmolStr,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}

pub fn file_symbols(file: &ast::File) -> Vec<FileSymbol> {
    let syntax = file.syntax();
    preorder(syntax.as_ref())
        .filter_map(to_symbol)
        .collect()
}

fn to_symbol(node: SyntaxNodeRef) -> Option<FileSymbol> {
    fn decl<'a, N: NameOwner<&'a SyntaxRoot>>(node: N) -> Option<FileSymbol> {
        let name = node.name()?;
        Some(FileSymbol {
            name: name.text(),
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


pub fn file_structure(file: &ast::File) -> Vec<StructureNode> {
    let mut res = Vec::new();
    let mut stack = Vec::new();
    let syntax = file.syntax();

    for event in walk(syntax.as_ref()) {
        match event {
            WalkEvent::Enter(node) => {
                match structure_node(node) {
                    Some(mut symbol) => {
                        symbol.parent = stack.last().map(|&n| n);
                        stack.push(res.len());
                        res.push(symbol);
                    }
                    None => (),
                }
            }
            WalkEvent::Exit(node) => {
                if structure_node(node).is_some() {
                    stack.pop().unwrap();
                }
            }
        }
    }
    res
}

fn structure_node(node: SyntaxNodeRef) -> Option<StructureNode> {
    fn decl<'a, N: NameOwner<&'a SyntaxRoot>>(node: N) -> Option<StructureNode> {
        let name = node.name()?;
        Some(StructureNode {
            parent: None,
            label: name.text().to_string(),
            navigation_range: name.syntax().range(),
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
        .visit(|im: ast::ImplItem<_>| {
            let mut label = String::new();
            let brace = im.syntax().children()
                .find(|it| {
                    let stop = it.kind() == L_CURLY;
                    if !stop {
                        label.push_str(&it.text());
                    }
                    stop
                })?;
            let navigation_range = TextRange::from_to(
                im.syntax().range().start(),
                brace.range().start(),
            );

            let node = StructureNode {
                parent: None,
                label,
                navigation_range,
                node_range: im.syntax().range(),
                kind: im.syntax().kind(),
            };
            Some(node)
        })
        .accept(node)?
}
