use ide_db::SymbolKind;
use syntax::{
    ast::{self, HasAttrs, HasGenericParams, HasName},
    match_ast, AstNode, AstToken, NodeOrToken, SourceFile, SyntaxNode, SyntaxToken, TextRange,
    WalkEvent,
};

#[derive(Debug, Clone)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: StructureNodeKind,
    pub detail: Option<String>,
    pub deprecated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StructureNodeKind {
    SymbolKind(SymbolKind),
    Region,
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
//
// image::https://user-images.githubusercontent.com/48062697/113020654-b42fc800-917a-11eb-8388-e7dc4d92b02e.gif[]

pub(crate) fn file_structure(file: &SourceFile) -> Vec<StructureNode> {
    let mut res = Vec::new();
    let mut stack = Vec::new();

    for event in file.syntax().preorder_with_tokens() {
        match event {
            WalkEvent::Enter(NodeOrToken::Node(node)) => {
                if let Some(mut symbol) = structure_node(&node) {
                    symbol.parent = stack.last().copied();
                    stack.push(res.len());
                    res.push(symbol);
                }
            }
            WalkEvent::Leave(NodeOrToken::Node(node)) => {
                if structure_node(&node).is_some() {
                    stack.pop().unwrap();
                }
            }
            WalkEvent::Enter(NodeOrToken::Token(token)) => {
                if let Some(mut symbol) = structure_token(token) {
                    symbol.parent = stack.last().copied();
                    stack.push(res.len());
                    res.push(symbol);
                }
            }
            WalkEvent::Leave(NodeOrToken::Token(token)) => {
                if structure_token(token).is_some() {
                    stack.pop().unwrap();
                }
            }
        }
    }
    res
}

fn structure_node(node: &SyntaxNode) -> Option<StructureNode> {
    fn decl<N: HasName + HasAttrs>(node: N, kind: StructureNodeKind) -> Option<StructureNode> {
        decl_with_detail(&node, None, kind)
    }

    fn decl_with_type_ref<N: HasName + HasAttrs>(
        node: &N,
        type_ref: Option<ast::Type>,
        kind: StructureNodeKind,
    ) -> Option<StructureNode> {
        let detail = type_ref.map(|type_ref| {
            let mut detail = String::new();
            collapse_ws(type_ref.syntax(), &mut detail);
            detail
        });
        decl_with_detail(node, detail, kind)
    }

    fn decl_with_detail<N: HasName + HasAttrs>(
        node: &N,
        detail: Option<String>,
        kind: StructureNodeKind,
    ) -> Option<StructureNode> {
        let name = node.name()?;

        Some(StructureNode {
            parent: None,
            label: name.text().to_string(),
            navigation_range: name.syntax().text_range(),
            node_range: node.syntax().text_range(),
            kind,
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
            ast::Fn(it) => {
                let mut detail = String::from("fn");
                if let Some(type_param_list) = it.generic_param_list() {
                    collapse_ws(type_param_list.syntax(), &mut detail);
                }
                if let Some(param_list) = it.param_list() {
                    collapse_ws(param_list.syntax(), &mut detail);
                }
                if let Some(ret_type) = it.ret_type() {
                    detail.push(' ');
                    collapse_ws(ret_type.syntax(), &mut detail);
                }

                decl_with_detail(&it, Some(detail), StructureNodeKind::SymbolKind(SymbolKind::Function))
            },
            ast::Struct(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Struct)),
            ast::Union(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Union)),
            ast::Enum(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Enum)),
            ast::Variant(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Variant)),
            ast::Trait(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Trait)),
            ast::TraitAlias(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::TraitAlias)),
            ast::Module(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Module)),
            ast::TypeAlias(it) => decl_with_type_ref(&it, it.ty(), StructureNodeKind::SymbolKind(SymbolKind::TypeAlias)),
            ast::RecordField(it) => decl_with_type_ref(&it, it.ty(), StructureNodeKind::SymbolKind(SymbolKind::Field)),
            ast::Const(it) => decl_with_type_ref(&it, it.ty(), StructureNodeKind::SymbolKind(SymbolKind::Const)),
            ast::Static(it) => decl_with_type_ref(&it, it.ty(), StructureNodeKind::SymbolKind(SymbolKind::Static)),
            ast::Impl(it) => {
                let target_type = it.self_ty()?;
                let target_trait = it.trait_();
                let label = match target_trait {
                    None => format!("impl {}", target_type.syntax().text()),
                    Some(t) => {
                        format!("impl {}{} for {}",
                            it.excl_token().map(|x| x.to_string()).unwrap_or_default(),
                            t.syntax().text(),
                            target_type.syntax().text(),
                        )
                    }
                };

                let node = StructureNode {
                    parent: None,
                    label,
                    navigation_range: target_type.syntax().text_range(),
                    node_range: it.syntax().text_range(),
                    kind: StructureNodeKind::SymbolKind(SymbolKind::Impl),
                    detail: None,
                    deprecated: false,
                };
                Some(node)
            },
            ast::Macro(it) => decl(it, StructureNodeKind::SymbolKind(SymbolKind::Macro)),
            _ => None,
        }
    }
}

fn structure_token(token: SyntaxToken) -> Option<StructureNode> {
    if let Some(comment) = ast::Comment::cast(token) {
        let text = comment.text().trim();

        if let Some(region_name) = text.strip_prefix("// region:").map(str::trim) {
            return Some(StructureNode {
                parent: None,
                label: region_name.to_string(),
                navigation_range: comment.syntax().text_range(),
                node_range: comment.syntax().text_range(),
                kind: StructureNodeKind::Region,
                detail: None,
                deprecated: false,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use super::*;

    fn check(ra_fixture: &str, expect: Expect) {
        let file = SourceFile::parse(ra_fixture).ok().unwrap();
        let structure = file_structure(&file);
        expect.assert_debug_eq(&structure)
    }

    #[test]
    fn test_negative_trait_bound() {
        let txt = r#"impl !Unpin for Test {}"#;
        check(
            txt,
            expect![[r#"
        [
            StructureNode {
                parent: None,
                label: "impl !Unpin for Test",
                navigation_range: 16..20,
                node_range: 0..23,
                kind: SymbolKind(
                    Impl,
                ),
                detail: None,
                deprecated: false,
            },
        ]
        "#]],
        );
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
trait Tr {}
trait Alias = Tr;

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

// region: Some region name
// endregion

// region: dontpanic
mod m {
fn f() {}
// endregion
fn g() {}
}
"#,
            expect![[r#"
                [
                    StructureNode {
                        parent: None,
                        label: "Foo",
                        navigation_range: 8..11,
                        node_range: 1..26,
                        kind: SymbolKind(
                            Struct,
                        ),
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
                        kind: SymbolKind(
                            Field,
                        ),
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
                        kind: SymbolKind(
                            Module,
                        ),
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
                        kind: SymbolKind(
                            Function,
                        ),
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
                        kind: SymbolKind(
                            Function,
                        ),
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
                        kind: SymbolKind(
                            Function,
                        ),
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
                        kind: SymbolKind(
                            Enum,
                        ),
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
                        kind: SymbolKind(
                            Variant,
                        ),
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
                        kind: SymbolKind(
                            Variant,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "T",
                        navigation_range: 186..187,
                        node_range: 181..193,
                        kind: SymbolKind(
                            TypeAlias,
                        ),
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
                        kind: SymbolKind(
                            Static,
                        ),
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
                        kind: SymbolKind(
                            Const,
                        ),
                        detail: Some(
                            "i32",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "Tr",
                        navigation_range: 239..241,
                        node_range: 233..244,
                        kind: SymbolKind(
                            Trait,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "Alias",
                        navigation_range: 251..256,
                        node_range: 245..262,
                        kind: SymbolKind(
                            TraitAlias,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "impl E",
                        navigation_range: 269..270,
                        node_range: 264..273,
                        kind: SymbolKind(
                            Impl,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "impl fmt::Debug for E",
                        navigation_range: 295..296,
                        node_range: 275..299,
                        kind: SymbolKind(
                            Impl,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mc",
                        navigation_range: 314..316,
                        node_range: 301..333,
                        kind: SymbolKind(
                            Macro,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mcexp",
                        navigation_range: 364..369,
                        node_range: 335..386,
                        kind: SymbolKind(
                            Macro,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "mcexp",
                        navigation_range: 417..422,
                        node_range: 388..439,
                        kind: SymbolKind(
                            Macro,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "obsolete",
                        navigation_range: 458..466,
                        node_range: 441..471,
                        kind: SymbolKind(
                            Function,
                        ),
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: true,
                    },
                    StructureNode {
                        parent: None,
                        label: "very_obsolete",
                        navigation_range: 511..524,
                        node_range: 473..529,
                        kind: SymbolKind(
                            Function,
                        ),
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: true,
                    },
                    StructureNode {
                        parent: None,
                        label: "Some region name",
                        navigation_range: 531..558,
                        node_range: 531..558,
                        kind: Region,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: None,
                        label: "m",
                        navigation_range: 598..599,
                        node_range: 573..636,
                        kind: SymbolKind(
                            Module,
                        ),
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            22,
                        ),
                        label: "dontpanic",
                        navigation_range: 573..593,
                        node_range: 573..593,
                        kind: Region,
                        detail: None,
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            22,
                        ),
                        label: "f",
                        navigation_range: 605..606,
                        node_range: 602..611,
                        kind: SymbolKind(
                            Function,
                        ),
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: false,
                    },
                    StructureNode {
                        parent: Some(
                            22,
                        ),
                        label: "g",
                        navigation_range: 628..629,
                        node_range: 612..634,
                        kind: SymbolKind(
                            Function,
                        ),
                        detail: Some(
                            "fn()",
                        ),
                        deprecated: false,
                    },
                ]
            "#]],
        );
    }
}
