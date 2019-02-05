use crate::TextRange;

use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, AttrsOwner, NameOwner, TypeParamsOwner},
    AstNode, SourceFile, SyntaxKind, SyntaxNode, WalkEvent,
};

#[derive(Debug, Clone)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: SyntaxKind,
    pub detail: Option<String>,
    pub deprecated: bool,
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
    fn decl<N: NameOwner + AttrsOwner>(node: &N) -> Option<StructureNode> {
        decl_with_detail(node, None)
    }

    fn decl_with_type_ref<N: NameOwner + AttrsOwner>(
        node: &N,
        type_ref: Option<&ast::TypeRef>,
    ) -> Option<StructureNode> {
        let detail = type_ref.map(|type_ref| {
            let mut detail = String::new();
            collapse_ws(type_ref.syntax(), &mut detail);
            detail
        });
        decl_with_detail(node, detail)
    }

    fn decl_with_detail<N: NameOwner + AttrsOwner>(
        node: &N,
        detail: Option<String>,
    ) -> Option<StructureNode> {
        let name = node.name()?;
        dbg!(name.text().to_string());
        Some(StructureNode {
            parent: None,
            label: name.text().to_string(),
            navigation_range: name.syntax().range(),
            node_range: node.syntax().range(),
            kind: node.syntax().kind(),
            detail,
            deprecated: node
                .attrs()
                .filter_map(|x| x.as_named())
                .any(|x| x == "deprecated"),
        })
    }

    fn collapse_ws(node: &SyntaxNode, output: &mut String) {
        let mut can_insert_ws = false;
        for line in node.text().chunks().flat_map(|chunk| chunk.lines()) {
            let line = line.trim();
            if line.is_empty() {
                if can_insert_ws {
                    output.push_str(" ");
                    can_insert_ws = false;
                }
            } else {
                output.push_str(line);
                can_insert_ws = true;
            }
        }
    }

    visitor()
        .visit(|fn_def: &ast::FnDef| {
            let mut detail = String::from("fn");
            if let Some(type_param_list) = fn_def.type_param_list() {
                collapse_ws(type_param_list.syntax(), &mut detail);
            }
            if let Some(param_list) = fn_def.param_list() {
                collapse_ws(param_list.syntax(), &mut detail);
            }
            if let Some(ret_type) = fn_def.ret_type() {
                detail.push_str(" ");
                collapse_ws(ret_type.syntax(), &mut detail);
            }

            decl_with_detail(fn_def, Some(detail))
        })
        .visit(decl::<ast::StructDef>)
        .visit(|nfd: &ast::NamedFieldDef| decl_with_type_ref(nfd, nfd.type_ref()))
        .visit(decl::<ast::EnumDef>)
        .visit(decl::<ast::EnumVariant>)
        .visit(decl::<ast::TraitDef>)
        .visit(decl::<ast::Module>)
        .visit(|td: &ast::TypeDef| decl_with_type_ref(td, td.type_ref()))
        .visit(|cd: &ast::ConstDef| decl_with_type_ref(cd, cd.type_ref()))
        .visit(|sd: &ast::StaticDef| decl_with_type_ref(sd, sd.type_ref()))
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
                detail: None,
                deprecated: false,
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
    fn bar1() {}
    fn bar2<T>(t: T) -> T {}
    fn bar3<A,
        B>(a: A,
        b: B) -> Vec<
        u32
    > {}
}

enum E { X, Y(i32) }
type T = ();
static S: i32 = 92;
const C: i32 = 92;

impl E {}

impl fmt::Debug for E {}

#[deprecated]
fn obsolete() {}

#[deprecated(note = "for awhile")]
fn very_obsolete() {}
"#,
        );
        let structure = file_structure(&file);
        assert_debug_snapshot_matches!("file_structure", structure);
    }
}
