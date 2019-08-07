use crate::{db::RootDatabase, FileId};
use hir::{HirDisplay, SourceAnalyzer, Ty};
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{
        AstNode, ForExpr, IfExpr, LambdaExpr, LetStmt, MatchArmList, Pat, PatKind, SourceFile,
        TypeAscriptionOwner, WhileExpr,
    },
    SmolStr, SyntaxKind, SyntaxNode, TextRange,
};

#[derive(Debug, PartialEq, Eq)]
pub enum InlayKind {
    TypeHint,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub kind: InlayKind,
    pub label: SmolStr,
}

pub(crate) fn inlay_hints(db: &RootDatabase, file_id: FileId, file: &SourceFile) -> Vec<InlayHint> {
    file.syntax()
        .descendants()
        .map(|node| get_inlay_hints(db, file_id, &node).unwrap_or_default())
        .flatten()
        .collect()
}

fn get_inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<Vec<InlayHint>> {
    visitor()
        .visit(|let_statement: LetStmt| {
            if let_statement.ascribed_type().is_some() {
                return None;
            }
            let pat = let_statement.pat()?;
            let analyzer = SourceAnalyzer::new(db, file_id, let_statement.syntax(), None);
            Some(get_pat_type_hints(db, &analyzer, pat, false))
        })
        .visit(|closure_parameter: LambdaExpr| {
            let analyzer = SourceAnalyzer::new(db, file_id, closure_parameter.syntax(), None);
            closure_parameter.param_list().map(|param_list| {
                param_list
                    .params()
                    .filter(|closure_param| closure_param.ascribed_type().is_none())
                    .filter_map(|closure_param| closure_param.pat())
                    .map(|root_pat| get_pat_type_hints(db, &analyzer, root_pat, false))
                    .flatten()
                    .collect()
            })
        })
        .visit(|for_expression: ForExpr| {
            let pat = for_expression.pat()?;
            let analyzer = SourceAnalyzer::new(db, file_id, for_expression.syntax(), None);
            Some(get_pat_type_hints(db, &analyzer, pat, false))
        })
        .visit(|if_expr: IfExpr| {
            let pat = if_expr.condition()?.pat()?;
            let analyzer = SourceAnalyzer::new(db, file_id, if_expr.syntax(), None);
            Some(get_pat_type_hints(db, &analyzer, pat, true))
        })
        .visit(|while_expr: WhileExpr| {
            let pat = while_expr.condition()?.pat()?;
            let analyzer = SourceAnalyzer::new(db, file_id, while_expr.syntax(), None);
            Some(get_pat_type_hints(db, &analyzer, pat, true))
        })
        .visit(|match_arm_list: MatchArmList| {
            let analyzer = SourceAnalyzer::new(db, file_id, match_arm_list.syntax(), None);
            Some(
                match_arm_list
                    .arms()
                    .map(|match_arm| match_arm.pats())
                    .flatten()
                    .map(|root_pat| get_pat_type_hints(db, &analyzer, root_pat, true))
                    .flatten()
                    .collect(),
            )
        })
        .accept(&node)?
}

fn get_pat_type_hints(
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    root_pat: Pat,
    skip_root_pat_hint: bool,
) -> Vec<InlayHint> {
    let original_pat = &root_pat.clone();

    get_leaf_pats(root_pat)
        .into_iter()
        .filter(|pat| !skip_root_pat_hint || pat != original_pat)
        .filter_map(|pat| {
            get_node_displayable_type(db, &analyzer, &pat)
                .map(|pat_type| (pat.syntax().text_range(), pat_type))
        })
        .map(|(range, pat_type)| InlayHint {
            range,
            kind: InlayKind::TypeHint,
            label: pat_type.display(db).to_string().into(),
        })
        .collect()
}

fn get_leaf_pats(root_pat: Pat) -> Vec<Pat> {
    let mut pats_to_process = std::collections::VecDeque::<Pat>::new();
    pats_to_process.push_back(root_pat);

    let mut leaf_pats = Vec::new();

    while let Some(maybe_leaf_pat) = pats_to_process.pop_front() {
        match maybe_leaf_pat.kind() {
            PatKind::BindPat(bind_pat) => {
                if let Some(pat) = bind_pat.pat() {
                    pats_to_process.push_back(pat);
                } else {
                    leaf_pats.push(maybe_leaf_pat);
                }
            }
            PatKind::TuplePat(tuple_pat) => {
                for arg_pat in tuple_pat.args() {
                    pats_to_process.push_back(arg_pat);
                }
            }
            PatKind::StructPat(struct_pat) => {
                if let Some(pat_list) = struct_pat.field_pat_list() {
                    pats_to_process.extend(
                        pat_list
                            .field_pats()
                            .filter_map(|field_pat| {
                                field_pat
                                    .pat()
                                    .filter(|pat| pat.syntax().kind() != SyntaxKind::BIND_PAT)
                            })
                            .chain(pat_list.bind_pats().map(|bind_pat| {
                                bind_pat.pat().unwrap_or_else(|| Pat::from(bind_pat))
                            })),
                    );
                }
            }
            PatKind::TupleStructPat(tuple_struct_pat) => {
                for arg_pat in tuple_struct_pat.args() {
                    pats_to_process.push_back(arg_pat);
                }
            }
            _ => (),
        }
    }
    leaf_pats
}

fn get_node_displayable_type(
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    node_pat: &Pat,
) -> Option<Ty> {
    analyzer.type_of_pat(db, node_pat).and_then(|resolved_type| {
        if let Ty::Apply(_) = resolved_type {
            Some(resolved_type)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::single_file;
    use insta::assert_debug_snapshot_matches;

    #[test]
    fn let_statement() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    struct InnerStruct {}

    let test = 54;
    let test: i32 = 33;
    let mut test = 33;
    let _ = 22;
    let test = "test";
    let test = InnerStruct {};

    let test = vec![222];
    let test: Vec<_> = (0..3).collect();
    let test = (0..3).collect::<Vec<i128>>();
    let test = (0..3).collect::<Vec<_>>();

    let mut test = Vec::new();
    test.push(333);

    let test = (42, 'a');
    let (a, (b, c, (d, e), f)) = (2, (3, 4, (6.6, 7.7), 5));
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [193; 197),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [236; 244),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [275; 279),
        kind: TypeHint,
        label: "&str",
    },
    InlayHint {
        range: [539; 543),
        kind: TypeHint,
        label: "(i32, char)",
    },
    InlayHint {
        range: [566; 567),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [570; 571),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [573; 574),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [584; 585),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [577; 578),
        kind: TypeHint,
        label: "f64",
    },
    InlayHint {
        range: [580; 581),
        kind: TypeHint,
        label: "f64",
    },
]"#
        );
    }

    #[test]
    fn closure_parameter() {
        let (analysis, file_id) = single_file(
            r#"
fn main() {
    let mut start = 0;
    (0..2).for_each(|increment| {
        start += increment;
    })
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [21; 30),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [57; 66),
        kind: TypeHint,
        label: "i32",
    },
]"#
        );
    }

    #[test]
    fn for_expression() {
        let (analysis, file_id) = single_file(
            r#"
fn main() {
    let mut start = 0;
    for increment in 0..2 {
        start += increment;
    }
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [21; 30),
        kind: TypeHint,
        label: "i32",
    },
    InlayHint {
        range: [44; 53),
        kind: TypeHint,
        label: "i32",
    },
]"#
        );
    }

    #[test]
    fn if_expr() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    let test = CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 });
    if let CustomOption::None = &test {};
    if let test = &test {};
    if let CustomOption::Some(test) = &test {};
    if let CustomOption::Some(Test { a, b }) = &test {};
    if let CustomOption::Some(Test { a: x, b: y }) = &test {};
    if let CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) = &test {};
    if let CustomOption::Some(Test { a: CustomOption::None, b: y }) = &test {};
    if let CustomOption::Some(Test { b: y, .. }) = &test {};

    if test == CustomOption::None {}
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [166; 170),
        kind: TypeHint,
        label: "CustomOption<Test>",
    },
    InlayHint {
        range: [334; 338),
        kind: TypeHint,
        label: "&Test",
    },
    InlayHint {
        range: [389; 390),
        kind: TypeHint,
        label: "&CustomOption<u32>",
    },
    InlayHint {
        range: [392; 393),
        kind: TypeHint,
        label: "&u8",
    },
    InlayHint {
        range: [531; 532),
        kind: TypeHint,
        label: "&u32",
    },
]"#
        );
    }

    #[test]
    fn while_expr() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    let test = CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 });
    while let CustomOption::None = &test {};
    while let test = &test {};
    while let CustomOption::Some(test) = &test {};
    while let CustomOption::Some(Test { a, b }) = &test {};
    while let CustomOption::Some(Test { a: x, b: y }) = &test {};
    while let CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) = &test {};
    while let CustomOption::Some(Test { a: CustomOption::None, b: y }) = &test {};
    while let CustomOption::Some(Test { b: y, .. }) = &test {};

    while test == CustomOption::None {}
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r###"
       ⋮[
       ⋮    InlayHint {
       ⋮        range: [166; 170),
       ⋮        kind: TypeHint,
       ⋮        label: "CustomOption<Test>",
       ⋮    },
       ⋮    InlayHint {
       ⋮        range: [343; 347),
       ⋮        kind: TypeHint,
       ⋮        label: "&Test",
       ⋮    },
       ⋮    InlayHint {
       ⋮        range: [401; 402),
       ⋮        kind: TypeHint,
       ⋮        label: "&CustomOption<u32>",
       ⋮    },
       ⋮    InlayHint {
       ⋮        range: [404; 405),
       ⋮        kind: TypeHint,
       ⋮        label: "&u8",
       ⋮    },
       ⋮    InlayHint {
       ⋮        range: [549; 550),
       ⋮        kind: TypeHint,
       ⋮        label: "&u32",
       ⋮    },
       ⋮]
        "###
        );
    }

    #[test]
    fn match_arm_list() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    match CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 }) {
        CustomOption::None => (),
        test => (),
        CustomOption::Some(test) => (),
        CustomOption::Some(Test { a, b }) => (),
        CustomOption::Some(Test { a: x, b: y }) => (),
        CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) => (),
        CustomOption::Some(Test { a: CustomOption::None, b: y }) => (),
        CustomOption::Some(Test { b: y, .. }) => (),
        _ => {}
    }
}"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [311; 315),
        kind: TypeHint,
        label: "Test",
    },
    InlayHint {
        range: [358; 359),
        kind: TypeHint,
        label: "CustomOption<u32>",
    },
    InlayHint {
        range: [361; 362),
        kind: TypeHint,
        label: "u8",
    },
    InlayHint {
        range: [484; 485),
        kind: TypeHint,
        label: "u32",
    },
]"#
        );
    }
}
