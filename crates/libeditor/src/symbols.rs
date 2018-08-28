use libsyntax2::{
    SyntaxKind, SyntaxNodeRef, AstNode, File, SmolStr,
    ast::{self, NameOwner},
    algo::{
        visit::{visitor, Visitor},
        walk::{walk, WalkEvent, preorder},
    },
};
use TextRange;

#[derive(Debug, Clone)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}

#[derive(Debug, Clone)]
pub struct FileSymbol {
    pub name: SmolStr,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
}

pub fn file_symbols(file: &File) -> Vec<FileSymbol> {
    preorder(file.syntax())
        .filter_map(to_symbol)
        .collect()
}

fn to_symbol(node: SyntaxNodeRef) -> Option<FileSymbol> {
    fn decl<'a, N: NameOwner<'a>>(node: N) -> Option<FileSymbol> {
        let name = node.name()?;
        Some(FileSymbol {
            name: name.text(),
            node_range: node.syntax().range(),
            kind: node.syntax().kind(),
        })
    }
    visitor()
        .visit(decl::<ast::FnDef>)
        .visit(decl::<ast::StructDef>)
        .visit(decl::<ast::EnumDef>)
        .visit(decl::<ast::TraitDef>)
        .visit(decl::<ast::Module>)
        .visit(decl::<ast::TypeDef>)
        .visit(decl::<ast::ConstDef>)
        .visit(decl::<ast::StaticDef>)
        .accept(node)?
}


pub fn file_structure(file: &File) -> Vec<StructureNode> {
    let mut res = Vec::new();
    let mut stack = Vec::new();

    for event in walk(file.syntax()) {
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
    fn decl<'a, N: NameOwner<'a>>(node: N) -> Option<StructureNode> {
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
        .visit(|im: ast::ImplItem| {
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
    use test_utils::assert_eq_dbg;

    #[test]
    fn test_file_structure() {
        let file = File::parse(r#"
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
"#);
        let symbols = file_structure(&file);
        assert_eq_dbg(
            r#"[StructureNode { parent: None, label: "Foo", navigation_range: [8; 11), node_range: [1; 26), kind: STRUCT_DEF },
                StructureNode { parent: Some(0), label: "x", navigation_range: [18; 19), node_range: [18; 24), kind: NAMED_FIELD_DEF },
                StructureNode { parent: None, label: "m", navigation_range: [32; 33), node_range: [28; 53), kind: MODULE },
                StructureNode { parent: Some(2), label: "bar", navigation_range: [43; 46), node_range: [40; 51), kind: FN_DEF },
                StructureNode { parent: None, label: "E", navigation_range: [60; 61), node_range: [55; 75), kind: ENUM_DEF },
                StructureNode { parent: None, label: "T", navigation_range: [81; 82), node_range: [76; 88), kind: TYPE_DEF },
                StructureNode { parent: None, label: "S", navigation_range: [96; 97), node_range: [89; 108), kind: STATIC_DEF },
                StructureNode { parent: None, label: "C", navigation_range: [115; 116), node_range: [109; 127), kind: CONST_DEF },
                StructureNode { parent: None, label: "impl E", navigation_range: [134; 135), node_range: [129; 138), kind: IMPL_ITEM },
                StructureNode { parent: None, label: "impl fmt::Debug for E", navigation_range: [160; 161), node_range: [140; 164), kind: IMPL_ITEM }]"#,
            &symbols,
        )
    }
}
