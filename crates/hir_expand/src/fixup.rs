use mbe::SyntheticToken;
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, AstNode},
    match_ast, SyntaxKind, SyntaxNode, SyntaxToken,
};
use tt::{Leaf, Subtree};

#[derive(Debug)]
pub struct SyntaxFixups {
    pub append: FxHashMap<SyntaxNode, Vec<SyntheticToken>>,
    pub replace: FxHashMap<SyntaxNode, Vec<SyntheticToken>>,
}

pub fn fixup_syntax(node: &SyntaxNode) -> SyntaxFixups {
    let mut append = FxHashMap::default();
    let mut replace = FxHashMap::default();
    let mut preorder = node.preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            syntax::WalkEvent::Enter(node) => node,
            syntax::WalkEvent::Leave(_) => continue,
        };
        if node.kind() == SyntaxKind::ERROR {
            // TODO this might not be helpful
            replace.insert(node, Vec::new());
            preorder.skip_subtree();
            continue;
        }
        match_ast! {
            match node {
                ast::FieldExpr(it) => {
                    if it.name_ref().is_none() {
                        // incomplete field access: some_expr.|
                        append.insert(node.clone(), vec![(SyntaxKind::IDENT, "__ra_fixup".into())]);
                    }
                },
                _ => (),
            }
        }
    }
    SyntaxFixups { append, replace }
}

pub fn reverse_fixups(tt: &mut Subtree) {
    tt.token_trees.retain(|tt| match tt {
        tt::TokenTree::Leaf(Leaf::Ident(ident)) => ident.text != "__ra_fixup",
        _ => true,
    });
    tt.token_trees.iter_mut().for_each(|tt| match tt {
        tt::TokenTree::Subtree(tt) => reverse_fixups(tt),
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::reverse_fixups;

    #[track_caller]
    fn check(ra_fixture: &str, mut expect: Expect) {
        let parsed = syntax::SourceFile::parse(ra_fixture);
        let fixups = super::fixup_syntax(&parsed.syntax_node());
        let (mut tt, _tmap) = mbe::syntax_node_to_token_tree_censored(
            &parsed.syntax_node(),
            fixups.replace,
            fixups.append,
        );

        let mut actual = tt.to_string();
        actual.push_str("\n");

        expect.indent(false);
        expect.assert_eq(&actual);

        // the fixed-up tree should be syntactically valid
        let (parse, _) = mbe::token_tree_to_syntax_node(&tt, ::mbe::TopEntryPoint::MacroItems);
        assert_eq!(parse.errors(), &[], "parse has syntax errors. parse tree:\n{:#?}", parse.syntax_node());

        reverse_fixups(&mut tt);

        // the fixed-up + reversed version should be equivalent to the original input
        // (but token IDs don't matter)
        let (original_as_tt, _) = mbe::syntax_node_to_token_tree(&parsed.syntax_node());
        assert_eq!(tt.to_string(), original_as_tt.to_string());
    }

    #[test]
    fn incomplete_field_expr_1() {
        check(r#"
fn foo() {
    a.
}
"#, expect![[r#"
fn foo () {a . __ra_fixup}
"#]])
    }

    #[test]
    fn incomplete_field_expr_2() {
        check(r#"
fn foo() {
    a. ;
}
"#, expect![[r#"
fn foo () {a . __ra_fixup ;}
"#]])
    }

    #[test]
    fn incomplete_field_expr_3() {
        check(r#"
fn foo() {
    a. ;
    bar();
}
"#, expect![[r#"
fn foo () {a . __ra_fixup ; bar () ;}
"#]])
    }

    #[test]
    fn field_expr_before_call() {
        // another case that easily happens while typing
        check(r#"
fn foo() {
    a.b
    bar();
}
"#, expect![[r#"
fn foo () {a . b bar () ;}
"#]])
    }
}
