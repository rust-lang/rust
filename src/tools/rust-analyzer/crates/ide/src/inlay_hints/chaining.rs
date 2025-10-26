//! Implementation of "chaining" inlay hints.
use hir::DisplayTarget;
use ide_db::famous_defs::FamousDefs;
use syntax::{
    Direction, NodeOrToken, SyntaxKind, T,
    ast::{self, AstNode},
};

use crate::{InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind};

use super::label_of_ty;

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    display_target: DisplayTarget,
    expr: &ast::Expr,
) -> Option<()> {
    if !config.chaining_hints {
        return None;
    }

    if matches!(expr, ast::Expr::RecordExpr(_)) {
        return None;
    }

    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);

    let mut tokens = expr
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter_map(NodeOrToken::into_token)
        .filter(|t| match t.kind() {
            SyntaxKind::WHITESPACE if !t.text().contains('\n') => false,
            SyntaxKind::COMMENT => false,
            _ => true,
        });

    // Chaining can be defined as an expression whose next sibling tokens are newline and dot
    // Ignoring extra whitespace and comments
    let next = tokens.next()?.kind();
    if next == SyntaxKind::WHITESPACE {
        let mut next_next = tokens.next()?.kind();
        while next_next == SyntaxKind::WHITESPACE {
            next_next = tokens.next()?.kind();
        }
        if next_next == T![.] {
            let ty = sema.type_of_expr(desc_expr)?.original;
            if ty.is_unknown() {
                return None;
            }
            if matches!(expr, ast::Expr::PathExpr(_))
                && let Some(hir::Adt::Struct(st)) = ty.as_adt()
                && st.fields(sema.db).is_empty()
            {
                return None;
            }
            let label = label_of_ty(famous_defs, config, &ty, display_target)?;
            acc.push(InlayHint {
                range: expr.syntax().text_range(),
                kind: InlayKind::Chaining,
                label,
                text_edit: None,
                position: InlayHintPosition::After,
                pad_left: true,
                pad_right: false,
                resolve_parent: Some(expr.syntax().text_range()),
            });
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use ide_db::text_edit::{TextRange, TextSize};

    use crate::{
        InlayHintsConfig, fixture,
        inlay_hints::{
            LazyProperty,
            tests::{DISABLED_CONFIG, TEST_CONFIG, check_expect, check_with_config},
        },
    };

    #[track_caller]
    fn check_chains(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[track_caller]
    pub(super) fn check_expect_clear_loc(
        config: InlayHintsConfig,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        expect: Expect,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let mut inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        inlay_hints.iter_mut().flat_map(|hint| &mut hint.label.parts).for_each(|hint| {
            if let Some(LazyProperty::Computed(loc)) = &mut hint.linked_location {
                loc.range = TextRange::empty(TextSize::from(0));
            }
        });
        let filtered =
            inlay_hints.into_iter().map(|hint| (hint.range, hint.label)).collect::<Vec<_>>();
        expect.assert_debug_eq(&filtered)
    }

    #[test]
    fn chaining_hints_ignore_comments() {
        check_expect(
            InlayHintsConfig { type_hints: false, chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C))
        .into_b() // This is a comment
        // This is another comment
        .into_c();
}
"#,
            expect![[r#"
                [
                    (
                        147..172,
                        [
                            InlayHintLabelPart {
                                text: "B",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 63..64,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        147..154,
                        [
                            InlayHintLabelPart {
                                text: "A",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 7..8,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn chaining_hints_without_newlines() {
        check_chains(
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C)).into_b().into_c();
}"#,
        );
    }

    #[test]
    fn disabled_location_links() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
    struct A { pub b: B }
    struct B { pub c: C }
    struct C(pub bool);
    struct D;

    impl D {
        fn foo(&self) -> i32 { 42 }
    }

    fn main() {
        let x = A { b: B { c: C(true) } }
            .b
            .c
            .0;
        let x = D
            .foo();
    }"#,
            expect![[r#"
                [
                    (
                        143..190,
                        [
                            InlayHintLabelPart {
                                text: "C",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 51..52,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        143..179,
                        [
                            InlayHintLabelPart {
                                text: "B",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 29..30,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn struct_access_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A { pub b: B }
struct B { pub c: C }
struct C(pub bool);
struct D;

impl D {
    fn foo(&self) -> i32 { 42 }
}

fn main() {
    let x = A { b: B { c: C(true) } }
        .b
        .c
        .0;
    let x = D
        .foo();
}"#,
            expect![[r#"
                [
                    (
                        143..190,
                        [
                            InlayHintLabelPart {
                                text: "C",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 51..52,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        143..179,
                        [
                            InlayHintLabelPart {
                                text: "B",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 29..30,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn generic_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A<T>(T);
struct B<T>(T);
struct C<T>(T);
struct X<T,R>(T, R);

impl<T> A<T> {
    fn new(t: T) -> Self { A(t) }
    fn into_b(self) -> B<T> { B(self.0) }
}
impl<T> B<T> {
    fn into_c(self) -> C<T> { C(self.0) }
}
fn main() {
    let c = A::new(X(42, true))
        .into_b()
        .into_c();
}
"#,
            expect![[r#"
                [
                    (
                        246..283,
                        [
                            InlayHintLabelPart {
                                text: "B",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 23..24,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<",
                            InlayHintLabelPart {
                                text: "X",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 55..56,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<i32, bool>>",
                        ],
                    ),
                    (
                        246..265,
                        [
                            InlayHintLabelPart {
                                text: "A",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 7..8,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<",
                            InlayHintLabelPart {
                                text: "X",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 55..56,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<i32, bool>>",
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn shorten_iterator_chaining_hints() {
        check_expect_clear_loc(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter.by_ref()
        .take(5)
        .by_ref()
        .take(5)
        .by_ref();
}
"#,
            expect![[r#"
                [
                    (
                        174..241,
                        [
                            "impl ",
                            InlayHintLabelPart {
                                text: "Iterator",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<",
                            InlayHintLabelPart {
                                text: "Item",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            " = ()>",
                        ],
                    ),
                    (
                        174..224,
                        [
                            "impl ",
                            InlayHintLabelPart {
                                text: "Iterator",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<",
                            InlayHintLabelPart {
                                text: "Item",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            " = ()>",
                        ],
                    ),
                    (
                        174..206,
                        [
                            "impl ",
                            InlayHintLabelPart {
                                text: "Iterator",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            "<",
                            InlayHintLabelPart {
                                text: "Item",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                            " = ()>",
                        ],
                    ),
                    (
                        174..189,
                        [
                            "&mut ",
                            InlayHintLabelPart {
                                text: "MyIter",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 0..0,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

    #[test]
    fn hints_in_attr_call() {
        check_expect(
            TEST_CONFIG,
            r#"
//- proc_macros: identity, input_replace
struct Struct;
impl Struct {
    fn chain(self) -> Self {
        self
    }
}
#[proc_macros::identity]
fn main() {
    let strukt = Struct;
    strukt
        .chain()
        .chain()
        .chain();
    Struct::chain(strukt);
}
"#,
            expect![[r#"
                [
                    (
                        124..130,
                        [
                            InlayHintLabelPart {
                                text: "Struct",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 7..13,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        145..185,
                        [
                            InlayHintLabelPart {
                                text: "Struct",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 7..13,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        145..168,
                        [
                            InlayHintLabelPart {
                                text: "Struct",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 7..13,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        222..228,
                        [
                            InlayHintLabelPart {
                                text: "self",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                0,
                                            ),
                                            range: 42..46,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }
}
