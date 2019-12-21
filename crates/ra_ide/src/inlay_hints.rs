//! FIXME: write short doc here

use hir::{HirDisplay, SourceAnalyzer};
use once_cell::unsync::Lazy;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, TypeAscriptionOwner},
    match_ast, SmolStr, SourceFile, SyntaxKind, SyntaxNode, TextRange,
};

use crate::{db::RootDatabase, FileId};

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

pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    file: &SourceFile,
    max_inlay_hint_length: Option<usize>,
) -> Vec<InlayHint> {
    file.syntax()
        .descendants()
        .flat_map(|node| get_inlay_hints(db, file_id, &node, max_inlay_hint_length))
        .flatten()
        .collect()
}

fn get_inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    node: &SyntaxNode,
    max_inlay_hint_length: Option<usize>,
) -> Option<Vec<InlayHint>> {
    let _p = profile("get_inlay_hints");
    let analyzer =
        Lazy::new(|| SourceAnalyzer::new(db, hir::InFile::new(file_id.into(), node), None));
    match_ast! {
        match node {
            ast::LetStmt(it) => {
                if it.ascribed_type().is_some() {
                    return None;
                }
                let pat = it.pat()?;
                Some(get_pat_type_hints(db, &analyzer, pat, false, max_inlay_hint_length))
            },
            ast::LambdaExpr(it) => {
                it.param_list().map(|param_list| {
                    param_list
                        .params()
                        .filter(|closure_param| closure_param.ascribed_type().is_none())
                        .filter_map(|closure_param| closure_param.pat())
                        .map(|root_pat| get_pat_type_hints(db, &analyzer, root_pat, false, max_inlay_hint_length))
                        .flatten()
                        .collect()
                })
            },
            ast::ForExpr(it) => {
                let pat = it.pat()?;
                Some(get_pat_type_hints(db, &analyzer, pat, false, max_inlay_hint_length))
            },
            ast::IfExpr(it) => {
                let pat = it.condition()?.pat()?;
                Some(get_pat_type_hints(db, &analyzer, pat, true, max_inlay_hint_length))
            },
            ast::WhileExpr(it) => {
                let pat = it.condition()?.pat()?;
                Some(get_pat_type_hints(db, &analyzer, pat, true, max_inlay_hint_length))
            },
            ast::MatchArmList(it) => {
                Some(
                    it
                        .arms()
                        .map(|match_arm| match_arm.pats())
                        .flatten()
                        .map(|root_pat| get_pat_type_hints(db, &analyzer, root_pat, true, max_inlay_hint_length))
                        .flatten()
                        .collect(),
                )
            },
            _ => None,
        }
    }
}

fn get_pat_type_hints(
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    root_pat: ast::Pat,
    skip_root_pat_hint: bool,
    max_inlay_hint_length: Option<usize>,
) -> Vec<InlayHint> {
    let original_pat = &root_pat.clone();

    get_leaf_pats(root_pat)
        .into_iter()
        .filter(|pat| !skip_root_pat_hint || pat != original_pat)
        .filter_map(|pat| {
            let ty = analyzer.type_of_pat(db, &pat)?;
            if ty.is_unknown() {
                return None;
            }
            Some((pat.syntax().text_range(), ty))
        })
        .map(|(range, pat_type)| InlayHint {
            range,
            kind: InlayKind::TypeHint,
            label: pat_type.display_truncated(db, max_inlay_hint_length).to_string().into(),
        })
        .collect()
}

fn get_leaf_pats(root_pat: ast::Pat) -> Vec<ast::Pat> {
    let mut pats_to_process = std::collections::VecDeque::<ast::Pat>::new();
    pats_to_process.push_back(root_pat);

    let mut leaf_pats = Vec::new();

    while let Some(maybe_leaf_pat) = pats_to_process.pop_front() {
        match &maybe_leaf_pat {
            ast::Pat::BindPat(bind_pat) => match bind_pat.pat() {
                Some(pat) => pats_to_process.push_back(pat),
                _ => leaf_pats.push(maybe_leaf_pat),
            },
            ast::Pat::TuplePat(tuple_pat) => pats_to_process.extend(tuple_pat.args()),
            ast::Pat::RecordPat(record_pat) => {
                if let Some(pat_list) = record_pat.record_field_pat_list() {
                    pats_to_process.extend(
                        pat_list
                            .record_field_pats()
                            .filter_map(|record_field_pat| {
                                record_field_pat
                                    .pat()
                                    .filter(|pat| pat.syntax().kind() != SyntaxKind::BIND_PAT)
                            })
                            .chain(pat_list.bind_pats().map(|bind_pat| {
                                bind_pat.pat().unwrap_or_else(|| ast::Pat::from(bind_pat))
                            })),
                    );
                }
            }
            ast::Pat::TupleStructPat(tuple_struct_pat) => {
                pats_to_process.extend(tuple_struct_pat.args())
            }
            ast::Pat::RefPat(ref_pat) => pats_to_process.extend(ref_pat.pat()),
            _ => (),
        }
    }
    leaf_pats
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::mock_analysis::single_file;

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        let (analysis, file_id) = single_file(
            r#"
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let zz = Test { t: 23, k: 33 };
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [69; 71),
                kind: TypeHint,
                label: "Test<i32>",
            },
        ]
        "###
        );
    }

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
    let &x = &92;
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
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
            InlayHint {
                range: [627; 628),
                kind: TypeHint,
                label: "i32",
            },
        ]
        "###
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

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
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
        ]
        "###
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

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
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
        ]
        "###
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

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
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
        ]
        "###
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

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [166; 170),
                kind: TypeHint,
                label: "CustomOption<Test>",
            },
            InlayHint {
                range: [343; 347),
                kind: TypeHint,
                label: "&Test",
            },
            InlayHint {
                range: [401; 402),
                kind: TypeHint,
                label: "&CustomOption<u32>",
            },
            InlayHint {
                range: [404; 405),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [549; 550),
                kind: TypeHint,
                label: "&u32",
            },
        ]
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

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
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
        ]
        "###
        );
    }

    #[test]
    fn hint_truncation() {
        let (analysis, file_id) = single_file(
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
    let b = VeryLongOuterName(0usize);
    let c = Smol(Smol(0u32))
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, Some(8)).unwrap(), @r###"
        [
            InlayHint {
                range: [74; 75),
                kind: TypeHint,
                label: "Smol<u32>",
            },
            InlayHint {
                range: [98; 99),
                kind: TypeHint,
                label: "VeryLongOuterName<…>",
            },
            InlayHint {
                range: [137; 138),
                kind: TypeHint,
                label: "Smol<Smol<…>>",
            },
        ]
        "###
        );
    }
}
