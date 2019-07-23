use crate::{db::RootDatabase, FileId};
use hir::{HirDisplay, Ty};
use ra_syntax::ast::Pat;
use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, PatKind, TypeAscriptionOwner},
    AstNode, SmolStr, SourceFile, SyntaxNode, TextRange,
};

#[derive(Debug, PartialEq, Eq)]
pub enum InlayKind {
    LetBindingType,
    ClosureParameterType,
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
            let let_syntax = let_statement.syntax();

            if let_statement.ascribed_type().is_some() {
                return None;
            }

            let let_pat = let_statement.pat()?;
            let inlay_type_string = get_node_displayable_type(db, file_id, let_syntax, &let_pat)?
                .display(db)
                .to_string()
                .into();

            let pat_range = match let_pat.kind() {
                PatKind::BindPat(bind_pat) => bind_pat.syntax().text_range(),
                PatKind::TuplePat(tuple_pat) => tuple_pat.syntax().text_range(),
                _ => return None,
            };

            Some(vec![InlayHint {
                range: pat_range,
                kind: InlayKind::LetBindingType,
                label: inlay_type_string,
            }])
        })
        .visit(|closure_parameter: ast::LambdaExpr| match closure_parameter.param_list() {
            Some(param_list) => Some(
                param_list
                    .params()
                    .filter(|closure_param| closure_param.ascribed_type().is_none())
                    .filter_map(|closure_param| {
                        let closure_param_syntax = closure_param.syntax();
                        let inlay_type_string = get_node_displayable_type(
                            db,
                            file_id,
                            closure_param_syntax,
                            &closure_param.pat()?,
                        )?
                        .display(db)
                        .to_string()
                        .into();

                        Some(InlayHint {
                            range: closure_param_syntax.text_range(),
                            kind: InlayKind::ClosureParameterType,
                            label: inlay_type_string,
                        })
                    })
                    .collect(),
            ),
            None => None,
        })
        .accept(&node)?
}

fn get_node_displayable_type(
    db: &RootDatabase,
    file_id: FileId,
    node_syntax: &SyntaxNode,
    node_pat: &Pat,
) -> Option<Ty> {
    let analyzer = hir::SourceAnalyzer::new(db, file_id, node_syntax, None);
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
    let test = InnerStruct {};
    let test = OuterStruct {};
    let test = vec![222];
    let mut test = Vec::new();
    test.push(333);
    let test = test.into_iter().map(|i| i * i).collect::<Vec<_>>();
    let mut test = 33;
    let _ = 22;
    let test: Vec<_> = (0..3).collect();

    let _ = (0..23).map(|i: u32| {
        let i_squared = i * i;
        i_squared
    });

    let test: i32 = 33;

    let (x, c) = (42, 'a');
    let test = (42, 'a');
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
        range: [121; 125),
        kind: LetBindingType,
        label: "OuterStruct",
    },
    InlayHint {
        range: [297; 305),
        kind: LetBindingType,
        label: "i32",
    },
    InlayHint {
        range: [417; 426),
        kind: LetBindingType,
        label: "u32",
    },
    InlayHint {
        range: [496; 502),
        kind: LetBindingType,
        label: "(i32, char)",
    },
    InlayHint {
        range: [524; 528),
        kind: LetBindingType,
        label: "(i32, char)",
    },
]"#
        );
    }
}
