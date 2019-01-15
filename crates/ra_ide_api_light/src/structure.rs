use crate::TextRange;

use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, NameOwner},
    AstNode, SourceFile, SyntaxKind, SyntaxNode, WalkEvent,
};

#[derive(Debug, Clone)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}

pub fn file_structure(file: &SourceFile) -> Vec<StructureNode> {
    let mut res = Vec::new();
    let mut stack = Vec::new();

    for event in file.syntax().preorder() {
        match event {
            WalkEvent::Enter(node) => {
                if let Some(mut symbol) = structure_node(node) {
                    symbol.parent = stack.last().map(|&n| n);
                    stack.push(res.len());
                    res.push(symbol);
                }
            }
            WalkEvent::Leave(node) => {
                if structure_node(node).is_some() {
                    stack.pop().unwrap();
                }
            }
        }
    }
    res
}

fn structure_node(node: &SyntaxNode) -> Option<StructureNode> {
    fn decl<N: NameOwner>(node: &N) -> Option<StructureNode> {
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
        .visit(decl::<ast::FnDef>)
        .visit(decl::<ast::StructDef>)
        .visit(decl::<ast::NamedFieldDef>)
        .visit(decl::<ast::EnumDef>)
        .visit(decl::<ast::TraitDef>)
        .visit(decl::<ast::Module>)
        .visit(decl::<ast::TypeDef>)
        .visit(decl::<ast::ConstDef>)
        .visit(decl::<ast::StaticDef>)
        .visit(|im: &ast::ImplBlock| {
            let target_type = im.target_type()?;
            let target_trait = im.target_trait();
            let label = match target_trait {
                None => format!("impl {}", target_type.syntax().text()),
                Some(t) => format!(
                    "impl {} for {}",
                    t.syntax().text(),
                    target_type.syntax().text(),
                ),
            };

            let node = StructureNode {
                parent: None,
                label,
                navigation_range: target_type.syntax().range(),
                node_range: im.syntax().range(),
                kind: im.syntax().kind(),
            };
            Some(node)
        })
        .accept(node)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_debug_snapshot_matches;

    #[test]
    fn test_file_structure() {
        let file = SourceFile::parse(
            r#"
struct Foo {
    x: i32
}

mod m {
    fn bar() {}
}

enum E { X, Y(i32) }
type T = ();
static S: i32 = 92;
const C: i32 = 92;

impl E {}

impl fmt::Debug for E {}
"#,
        );
        let structure = file_structure(&file);
        assert_debug_snapshot_matches!("file_structure", structure);
    }
}
