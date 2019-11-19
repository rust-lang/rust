//! This modules implements "expand macro" functionality in the IDE

use crate::{db::RootDatabase, FilePosition};
use hir::db::AstDatabase;
use ra_db::SourceDatabase;
use rustc_hash::FxHashMap;

use ra_syntax::{
    algo::{find_node_at_offset, replace_descendants},
    ast::{self},
    AstNode, NodeOrToken, SyntaxKind, SyntaxNode, WalkEvent, T,
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

// FIXME: It would also be cool to share logic here and in the mbe tests,
// which are pretty unreadable at the moment.
fn insert_whitespaces(syn: SyntaxNode) -> String {
    use SyntaxKind::*;

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

    let mut indent = 0;
    let mut last: Option<SyntaxKind> = None;

    while let Some(token) = token_iter.next() {
        let mut is_next = |f: fn(SyntaxKind) -> bool, default| -> bool {
            token_iter.peek().map(|it| f(it.kind())).unwrap_or(default)
        };
        let is_last = |f: fn(SyntaxKind) -> bool, default| -> bool {
            last.map(|it| f(it)).unwrap_or(default)
        };

        res += &match token.kind() {
            k @ _
                if (k.is_keyword() || k.is_literal() || k == IDENT)
                    && is_next(|it| !it.is_punct(), true) =>
            {
                token.text().to_string() + " "
            }
            L_CURLY if is_next(|it| it != R_CURLY, true) => {
                indent += 1;
                format!(" {{\n{}", "  ".repeat(indent))
            }
            R_CURLY if is_last(|it| it != L_CURLY, true) => {
                indent = indent.checked_sub(1).unwrap_or(0);
                format!("\n}}{}", "  ".repeat(indent))
            }
            R_CURLY => {
                indent = indent.checked_sub(1).unwrap_or(0);
                format!("}}\n{}", "  ".repeat(indent))
            }
            T![;] => format!(";\n{}", "  ".repeat(indent)),
            T![->] => " -> ".to_string(),
            T![=] => " = ".to_string(),
            T![=>] => " => ".to_string(),
            _ => token.text().to_string(),
        };

        last = Some(token.kind());
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_analysis::analysis_and_position;
    use insta::assert_snapshot;

    fn check_expand_macro(fixture: &str) -> ExpandedMacro {
        let (analysis, pos) = analysis_and_position(fixture);
        analysis.expand_macro(pos).unwrap().unwrap()
    }

    #[test]
    fn macro_expand_recursive_expansion() {
        let res = check_expand_macro(
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
        );

        assert_eq!(res.name, "foo");
        assert_snapshot!(res.expansion, @r###"
fn b(){}
"###);
    }

    #[test]
    fn macro_expand_multiple_lines() {
        let res = check_expand_macro(
            r#"
        //- /lib.rs
        macro_rules! foo {
            () => { 
                fn some_thing() -> u32 {
                    let a = 0;
                    a + 10
                }
            }
        }
        f<|>oo!();
        "#,
        );

        assert_eq!(res.name, "foo");
        assert_snapshot!(res.expansion, @r###"
fn some_thing() -> u32 {
  let a = 0;
  a+10
}        
"###);
    }
}
