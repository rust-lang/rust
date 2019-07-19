use ra_syntax::{
    algo::visit::{visitor, Visitor},
    ast::{self, PatKind, TypeAscriptionOwner},
    AstNode, SmolStr, SourceFile, SyntaxNode, TextRange,
};

#[derive(Debug, PartialEq, Eq)]
pub enum InlayKind {
    LetBinding,
    ClosureParameter,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub text: SmolStr,
    pub inlay_kind: InlayKind,
}

pub(crate) fn inlay_hints(file: &SourceFile) -> Vec<InlayHint> {
    file.syntax().descendants().map(|node| get_inlay_hints(&node)).flatten().collect()
}

fn get_inlay_hints(node: &SyntaxNode) -> Vec<InlayHint> {
    visitor()
        .visit(|let_statement: ast::LetStmt| {
            let let_syntax = let_statement.syntax();

            if let_statement.ascribed_type().is_some() {
                return Vec::new();
            }

            let pat_range = match let_statement.pat().map(|pat| pat.kind()) {
                Some(PatKind::BindPat(bind_pat)) => bind_pat.syntax().text_range(),
                Some(PatKind::TuplePat(tuple_pat)) => tuple_pat.syntax().text_range(),
                _ => return Vec::new(),
            };

            vec![InlayHint {
                range: pat_range,
                text: let_syntax.text().to_smol_string(),
                inlay_kind: InlayKind::LetBinding,
            }]
        })
        .visit(|closure_parameter: ast::LambdaExpr| {
            if let Some(param_list) = closure_parameter.param_list() {
                param_list
                    .params()
                    .filter(|closure_param| closure_param.ascribed_type().is_none())
                    .map(|closure_param| {
                        let closure_param_syntax = closure_param.syntax();
                        InlayHint {
                            range: closure_param_syntax.text_range(),
                            text: closure_param_syntax.text().to_smol_string(),
                            inlay_kind: InlayKind::ClosureParameter,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })
        .accept(&node)
        .unwrap_or_else(Vec::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_debug_snapshot_matches;

    #[test]
    fn test_inlay_hints() {
        let file = SourceFile::parse(
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
        )
        .ok()
        .unwrap();
        let hints = inlay_hints(&file);
        assert_debug_snapshot_matches!("inlay_hints", hints);
    }
}
