use ra_syntax::{
    ast::{self, AttrsOwner, NameOwner, TypeAscriptionOwner, TypeParamsOwner},
    match_ast, AstNode, SourceFile, SyntaxKind, SyntaxNode, TextRange, WalkEvent,
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

// Feature: File Structure
//
// Provides a tree of the symbols defined in the file. Can be used to
//
// * fuzzy search symbol in a file (super useful)
// * draw breadcrumbs to describe the context around the cursor
// * draw outline of the file
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Ctrl+Shift+O]
// |===
pub fn file_structure(file: &SourceFile) -> Vec<StructureNode> {
    let mut res = Vec::new();
    let mut stack = Vec::new();

    for event in file.syntax().preorder() {
        match event {
            WalkEvent::Enter(node) => {
                if let Some(mut symbol) = structure_node(&node) {
                    symbol.parent = stack.last().copied();
                    stack.push(res.len());
                    res.push(symbol);
                }
            }
            WalkEvent::Leave(node) => {
                if structure_node(&node).is_some() {
                    stack.pop().unwrap();
                }
            }
        }
    }
    res
}

fn structure_node(node: &SyntaxNode) -> Option<StructureNode> {
    fn decl<N: NameOwner + AttrsOwner>(node: N) -> Option<StructureNode> {
        decl_with_detail(node, None)
    }

    fn decl_with_ascription<N: NameOwner + AttrsOwner + TypeAscriptionOwner>(
        node: N,
    ) -> Option<StructureNode> {
        let ty = node.ascribed_type();
        decl_with_type_ref(node, ty)
    }

    fn decl_with_type_ref<N: NameOwner + AttrsOwner>(
        node: N,
        type_ref: Option<ast::TypeRef>,
    ) -> Option<StructureNode> {
        let detail = type_ref.map(|type_ref| {
            let mut detail = String::new();
            collapse_ws(type_ref.syntax(), &mut detail);
            detail
        });
        decl_with_detail(node, detail)
    }

    fn decl_with_detail<N: NameOwner + AttrsOwner>(
        node: N,
        detail: Option<String>,
    ) -> Option<StructureNode> {
        let name = node.name()?;

        Some(StructureNode {
            parent: None,
            label: name.text().to_string(),
            navigation_range: name.syntax().text_range(),
            node_range: node.syntax().text_range(),
            kind: node.syntax().kind(),
            detail,
            deprecated: node.attrs().filter_map(|x| x.simple_name()).any(|x| x == "deprecated"),
        })
    }

    fn collapse_ws(node: &SyntaxNode, output: &mut String) {
        let mut can_insert_ws = false;
        node.text().for_each_chunk(|chunk| {
            for line in chunk.lines() {
                let line = line.trim();
                if line.is_empty() {
                    if can_insert_ws {
                        output.push(' ');
                        can_insert_ws = false;
                    }
                } else {
                    output.push_str(line);
                    can_insert_ws = true;
                }
            }
        })
    }

    match_ast! {
        match node {
            ast::FnDef(it) => {
                let mut detail = String::from("fn");
                if let Some(type_param_list) = it.type_param_list() {
                    collapse_ws(type_param_list.syntax(), &mut detail);
                }
                if let Some(param_list) = it.param_list() {
                    collapse_ws(param_list.syntax(), &mut detail);
                }
                if let Some(ret_type) = it.ret_type() {
                    detail.push_str(" ");
                    collapse_ws(ret_type.syntax(), &mut detail);
                }

                decl_with_detail(it, Some(detail))
            },
            ast::StructDef(it) => decl(it),
            ast::EnumDef(it) => decl(it),
            ast::EnumVariant(it) => decl(it),
            ast::TraitDef(it) => decl(it),
            ast::Module(it) => decl(it),
            ast::TypeAliasDef(it) => {
                let ty = it.type_ref();
                decl_with_type_ref(it, ty)
            },
            ast::RecordFieldDef(it) => decl_with_ascription(it),
            ast::ConstDef(it) => decl_with_ascription(it),
            ast::StaticDef(it) => decl_with_ascription(it),
            ast::ImplDef(it) => {
                let target_type = it.target_type()?;
                let target_trait = it.target_trait();
                let label = match target_trait {
                    None => format!("impl {}", target_type.syntax().text()),
                    Some(t) => {
                        format!("impl {} for {}", t.syntax().text(), target_type.syntax().text(),)
                    }
                };

                let node = StructureNode {
                    parent: None,
                    label,
                    navigation_range: target_type.syntax().text_range(),
                    node_range: it.syntax().text_range(),
                    kind: it.syntax().kind(),
                    detail: None,
                    deprecated: false,
                };
                Some(node)
            },
            ast::MacroCall(it) => {
                match it.path().and_then(|it| it.segment()).and_then(|it| it.name_ref()) {
                    Some(path_segment) if path_segment.text() == "macro_rules"
                    => decl(it),
                    _ => None,
                }
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use super::*;

    fn check(ra_fixture: &str, expect: Expect) {
        let file = SourceFile::parse(ra_fixture).ok().unwrap();
        let structure = file_structure(&file);
        expect.assert_debug_eq(&structure)
    }

    #[test]
    fn test_file_structure() {
        check(
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

macro_rules! mc {
    () => {}
}

#[macro_export]
macro_rules! mcexp {
    () => {}
}

/// Doc comment
macro_rules! mcexp {
    () => {}
}

#[deprecated]
fn obsolete() {}

#[deprecated(note = "for awhile")]
fn very_obsolete() {}
"#,
            expect![[r#"
                [
                    StructureNode {
                        parent: None,
                        label: "Foo",
                        navigation_range: 8..11,
                        node_range: 1..26,
                        kind: STRUCT_DEF,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            0,
                        ),
                        label: "x",
                        navigation_range: 18..19,
                        node_range: 18..24,
                        kind: RECORD_FIELD_DEF,
                        detail: Some(
                            "i32",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "m",
                        navigation_range: 32..33,
                        node_range: 28..158,
                        kind: MODULE,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            2,
                        ),
                        label: "bar1",
                        navigation_range: 43..47,
                        node_range: 40..52,
                        kind: FN_DEF,
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            2,
                        ),
                        label: "bar2",
                        navigation_range: 60..64,
                        node_range: 57..81,
                        kind: FN_DEF,
                        detail: Some(
                            "fn<T>(t: T) -> T",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            2,
                        ),
                        label: "bar3",
                        navigation_range: 89..93,
                        node_range: 86..156,
                        kind: FN_DEF,
                        detail: Some(
                            "fn<A, B>(a: A, b: B) -> Vec< u32 >",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "E",
                        navigation_range: 165..166,
                        node_range: 160..180,
                        kind: ENUM_DEF,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            6,
                        ),
                        label: "X",
                        navigation_range: 169..170,
                        node_range: 169..170,
                        kind: ENUM_VARIANT,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            6,
                        ),
                        label: "Y",
                        navigation_range: 172..173,
                        node_range: 172..178,
                        kind: ENUM_VARIANT,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "T",
                        navigation_range: 186..187,
                        node_range: 181..193,
                        kind: TYPE_ALIAS_DEF,
                        detail: Some(
                            "()",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "S",
                        navigation_range: 201..202,
                        node_range: 194..213,
                        kind: STATIC_DEF,
                        detail: Some(
                            "i32",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "C",
                        navigation_range: 220..221,
                        node_range: 214..232,
                        kind: CONST_DEF,
                        detail: Some(
                            "i32",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "impl E",
                        navigation_range: 239..240,
                        node_range: 234..243,
                        kind: IMPL_DEF,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "impl fmt::Debug for E",
                        navigation_range: 265..266,
                        node_range: 245..269,
                        kind: IMPL_DEF,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mc",
                        navigation_range: 284..286,
                        node_range: 271..303,
                        kind: MACRO_CALL,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mcexp",
                        navigation_range: 334..339,
                        node_range: 305..356,
                        kind: MACRO_CALL,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mcexp",
                        navigation_range: 387..392,
                        node_range: 358..409,
                        kind: MACRO_CALL,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "obsolete",
                        navigation_range: 428..436,
                        node_range: 411..441,
                        kind: FN_DEF,
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: true,
                    },
                    StructureNode {
                        parent: None,
                        label: "very_obsolete",
                        navigation_range: 481..494,
                        node_range: 443..499,
                        kind: FN_DEF,
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: true,
                    },
                ]
            "#]],
        );
    }
}
