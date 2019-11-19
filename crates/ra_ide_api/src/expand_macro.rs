//! FIXME: write short doc here

use crate::{db::RootDatabase, FilePosition};
use hir::db::AstDatabase;
use ra_db::SourceDatabase;
use rustc_hash::FxHashMap;

use ra_syntax::{
    algo::{find_node_at_offset, replace_descendants},
    ast::{self},
    AstNode, NodeOrToken, SyntaxKind, SyntaxNode, WalkEvent,
};

pub struct ExpandedMacro {
    pub name: String,
    pub expansion: String,
}

pub(crate) fn expand_macro(db: &RootDatabase, position: FilePosition) -> Option<ExpandedMacro> {
    let parse = db.parse(position.file_id);
    let file = parse.tree();
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset)?;
    let mac = name_ref.syntax().ancestors().find_map(ast::MacroCall::cast)?;

    let source = hir::Source::new(position.file_id.into(), mac.syntax());
    let expanded = expand_macro_recur(db, source, &mac)?;

    // FIXME:
    // macro expansion may lose all white space information
    // But we hope someday we can use ra_fmt for that
    let expansion = insert_whitespaces(expanded);
    Some(ExpandedMacro { name: name_ref.text().to_string(), expansion })
}

fn expand_macro_recur(
    db: &RootDatabase,
    source: hir::Source<&SyntaxNode>,
    macro_call: &ast::MacroCall,
) -> Option<SyntaxNode> {
    let analyzer = hir::SourceAnalyzer::new(db, source, None);
    let expansion = analyzer.expand(db, &macro_call)?;
    let macro_file_id = expansion.file_id();
    let expanded: SyntaxNode = db.parse_or_expand(macro_file_id)?;

    let children = expanded.descendants().filter_map(ast::MacroCall::cast);
    let mut replaces = FxHashMap::default();

    for child in children.into_iter() {
        let source = hir::Source::new(macro_file_id, source.ast);
        let new_node = expand_macro_recur(db, source, &child)?;

        replaces.insert(child.syntax().clone().into(), new_node.into());
    }

    Some(replace_descendants(&expanded, &replaces))
}

fn insert_whitespaces(syn: SyntaxNode) -> String {
    let mut res = String::new();

    let mut token_iter = syn
        .preorder_with_tokens()
        .filter_map(|event| {
            if let WalkEvent::Enter(NodeOrToken::Token(token)) = event {
                Some(token)
            } else {
                None
            }
        })
        .peekable();

    while let Some(token) = token_iter.next() {
        res += &token.text().to_string();
        if token.kind().is_keyword()
            || token.kind().is_literal()
            || token.kind() == SyntaxKind::IDENT
        {
            if !token_iter.peek().map(|it| it.kind().is_punct()).unwrap_or(false) {
                res += " ";
            }
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::analysis_and_position;

    fn check_expand_macro(fixture: &str, expected: (&str, &str)) {
        let (analysis, pos) = analysis_and_position(fixture);

        let result = analysis.expand_macro(pos).unwrap().unwrap();
        assert_eq!(result.name, expected.0.to_string());
        assert_eq!(result.expansion, expected.1.to_string());
    }

    #[test]
    fn macro_expand_recursive_expansion() {
        check_expand_macro(
            r#"
        //- /lib.rs
        macro_rules! bar {
            () => { fn  b() {} }
        }
        macro_rules! foo {
            () => { bar!(); }
        }
        macro_rules! baz {
            () => { foo!(); }
        }        
        f<|>oo!();
        "#,
            ("foo", "fn b(){}"),
        );
    }
}
