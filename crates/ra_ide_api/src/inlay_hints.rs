use crate::{db::RootDatabase, FileId};
use hir::{HirDisplay, SourceAnalyzer, Ty};
use ra_syntax::ast::Pat;
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, PatKind, TypeAscriptionOwner},
    AstNode, SmolStr, SourceFile, SyntaxNode, TextRange,
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum InlayKind {
    LetBindingType,
    ClosureParameterType,
    ForExpressionBindingType,
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
        .visit(|let_statement: ast::LetStmt| {
            if let_statement.ascribed_type().is_some() {
                return None;
            }
            let analyzer = SourceAnalyzer::new(db, file_id, let_statement.syntax(), None);
            Some(get_pat_hints(db, &analyzer, let_statement.pat()?, InlayKind::LetBindingType))
        })
        .visit(|closure_parameter: ast::LambdaExpr| {
            let analyzer = SourceAnalyzer::new(db, file_id, closure_parameter.syntax(), None);
            closure_parameter.param_list().map(|param_list| {
                param_list
                    .params()
                    .filter(|closure_param| closure_param.ascribed_type().is_none())
                    .filter_map(|closure_param| closure_param.pat())
                    .map(|root_pat| {
                        get_pat_hints(db, &analyzer, root_pat, InlayKind::ClosureParameterType)
                    })
                    .flatten()
                    .collect()
            })
        })
        .visit(|for_expression: ast::ForExpr| {
            let analyzer = SourceAnalyzer::new(db, file_id, for_expression.syntax(), None);
            Some(get_pat_hints(
                db,
                &analyzer,
                for_expression.pat()?,
                InlayKind::ForExpressionBindingType,
            ))
        })
        .accept(&node)?
}

fn get_pat_hints(
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    root_pat: Pat,
    kind: InlayKind,
) -> Vec<InlayHint> {
    get_leaf_pats(root_pat)
        .into_iter()
        .filter_map(|pat| {
            get_node_displayable_type(db, &analyzer, &pat)
                .map(|pat_type| (pat.syntax().text_range(), pat_type))
        })
        .map(|(range, pat_type)| InlayHint {
            range,
            kind: kind.clone(),
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
    fn test_inlay_hints() {
        let (analysis, file_id) = single_file(
            r#"
struct OuterStruct {}

fn main() {
    struct InnerStruct {}

    let test = 54;
    let test: i32 = 33;
    let mut test = 33;
    let _ = 22;
    let test = "test";
    let test = InnerStruct {};
    let test = OuterStruct {};

    let test = vec![222];
    let test: Vec<_> = (0..3).collect();

    let mut test = Vec::new();
    test.push(333);

    let test = test.into_iter().map(|i| i * i).collect::<Vec<_>>();
    let test = test.into_iter().map(|i| i * i).collect::<Vec<u128>>();

    let _ = (0..23).map(|i: u32| {
        let i_squared = i * i;
        i_squared
    });

    let test = (42, 'a');
    let (a, (b, c, (d, e), f)) = (2, (3, 4, (6.6, 7.7), 5));

    let test = Some((2, 3));
    for (i, j) in test {}
}
"#,
        );

        assert_debug_snapshot_matches!(analysis.inlay_hints(file_id).unwrap(), @r#"[
    InlayHint {
        range: [71; 75),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [114; 122),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [153; 157),
        kind: LetBindingType,
        label: "&str",
    },
    InlayHint {
        range: [207; 211),
        kind: LetBindingType,
        label: "OuterStruct",
    },
    InlayHint {
        range: [538; 547),
        kind: LetBindingType,
        label: "u32",
    },
    InlayHint {
        range: [592; 596),
        kind: LetBindingType,
        label: "(i32, char)",
    },
    InlayHint {
        range: [619; 620),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [623; 624),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [626; 627),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [637; 638),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [630; 631),
        kind: LetBindingType,
        label: "f64",
    },
    InlayHint {
        range: [633; 634),
        kind: LetBindingType,
        label: "f64",
    },
]"#
        );
    }
}
