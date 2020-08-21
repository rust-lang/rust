use hir::Semantics;
use ide_db::RootDatabase;
use syntax::{
    algo::{find_node_at_offset, SyntaxRewriter},
    ast, AstNode, NodeOrToken, SyntaxKind,
    SyntaxKind::*,
    SyntaxNode, WalkEvent, T,
};

use crate::FilePosition;

pub struct ExpandedMacro {
    pub name: String,
    pub expansion: String,
}

// Feature: Expand Macro Recursively
//
// Shows the full macro expansion of the macro at current cursor.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Expand macro recursively**
// |===
pub(crate) fn expand_macro(db: &RootDatabase, position: FilePosition) -> Option<ExpandedMacro> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id);
    let name_ref = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset)?;
    let mac = name_ref.syntax().ancestors().find_map(ast::MacroCall::cast)?;

    let expanded = expand_macro_recur(&sema, &mac)?;

    // FIXME:
    // macro expansion may lose all white space information
    // But we hope someday we can use ra_fmt for that
    let expansion = insert_whitespaces(expanded);
    Some(ExpandedMacro { name: name_ref.text().to_string(), expansion })
}

fn expand_macro_recur(
    sema: &Semantics<RootDatabase>,
    macro_call: &ast::MacroCall,
) -> Option<SyntaxNode> {
    let mut expanded = sema.expand(macro_call)?;

    let children = expanded.descendants().filter_map(ast::MacroCall::cast);
    let mut rewriter = SyntaxRewriter::default();

    for child in children.into_iter() {
        if let Some(new_node) = expand_macro_recur(sema, &child) {
            // Replace the whole node if it is root
            // `replace_descendants` will not replace the parent node
            // but `SyntaxNode::descendants include itself
            if expanded == *child.syntax() {
                expanded = new_node;
            } else {
                rewriter.replace(child.syntax(), &new_node)
            }
        }
    }

    let res = rewriter.rewrite(&expanded);
    Some(res)
}

// FIXME: It would also be cool to share logic here and in the mbe tests,
// which are pretty unreadable at the moment.
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

    let mut indent = 0;
    let mut last: Option<SyntaxKind> = None;

    while let Some(token) = token_iter.next() {
        let mut is_next = |f: fn(SyntaxKind) -> bool, default| -> bool {
            token_iter.peek().map(|it| f(it.kind())).unwrap_or(default)
        };
        let is_last =
            |f: fn(SyntaxKind) -> bool, default| -> bool { last.map(f).unwrap_or(default) };

        res += &match token.kind() {
            k if is_text(k) && is_next(|it| !it.is_punct(), true) => token.text().to_string() + " ",
            L_CURLY if is_next(|it| it != R_CURLY, true) => {
                indent += 1;
                let leading_space = if is_last(is_text, false) { " " } else { "" };
                format!("{}{{\n{}", leading_space, "  ".repeat(indent))
            }
            R_CURLY if is_last(|it| it != L_CURLY, true) => {
                indent = indent.saturating_sub(1);
                format!("\n{}}}", "  ".repeat(indent))
            }
            R_CURLY => format!("}}\n{}", "  ".repeat(indent)),
            T![;] => format!(";\n{}", "  ".repeat(indent)),
            T![->] => " -> ".to_string(),
            T![=] => " = ".to_string(),
            T![=>] => " => ".to_string(),
            _ => token.text().to_string(),
        };

        last = Some(token.kind());
    }

    return res;

    fn is_text(k: SyntaxKind) -> bool {
        k.is_keyword() || k.is_literal() || k == IDENT
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::mock_analysis::analysis_and_position;

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, pos) = analysis_and_position(ra_fixture);
        let expansion = analysis.expand_macro(pos).unwrap().unwrap();
        let actual = format!("{}\n{}", expansion.name, expansion.expansion);
        expect.assert_eq(&actual);
    }

    #[test]
    fn macro_expand_recursive_expansion() {
        check(
            r#"
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
            expect![[r#"
                foo
                fn b(){}
            "#]],
        );
    }

    #[test]
    fn macro_expand_multiple_lines() {
        check(
            r#"
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
            expect![[r#"
            foo
            fn some_thing() -> u32 {
              let a = 0;
              a+10
            }"#]],
        );
    }

    #[test]
    fn macro_expand_match_ast() {
        check(
            r#"
macro_rules! match_ast {
    (match $node:ident { $($tt:tt)* }) => { match_ast!(match ($node) { $($tt)* }) };
    (match ($node:expr) {
        $( ast::$ast:ident($it:ident) => $res:block, )*
        _ => $catch_all:expr $(,)?
    }) => {{
        $( if let Some($it) = ast::$ast::cast($node.clone()) $res else )*
        { $catch_all }
    }};
}

fn main() {
    mat<|>ch_ast! {
        match container {
            ast::TraitDef(it) => {},
            ast::ImplDef(it) => {},
            _ => { continue },
        }
    }
}
"#,
            expect![[r#"
       match_ast
       {
         if let Some(it) = ast::TraitDef::cast(container.clone()){}
         else if let Some(it) = ast::ImplDef::cast(container.clone()){}
         else {
           {
             continue
           }
         }
       }"#]],
        );
    }

    #[test]
    fn macro_expand_match_ast_inside_let_statement() {
        check(
            r#"
macro_rules! match_ast {
    (match $node:ident { $($tt:tt)* }) => { match_ast!(match ($node) { $($tt)* }) };
    (match ($node:expr) {}) => {{}};
}

fn main() {
    let p = f(|it| {
        let res = mat<|>ch_ast! { match c {}};
        Some(res)
    })?;
}
"#,
            expect![[r#"
                match_ast
                {}
            "#]],
        );
    }

    #[test]
    fn macro_expand_inner_macro_fail_to_expand() {
        check(
            r#"
macro_rules! bar {
    (BAD) => {};
}
macro_rules! foo {
    () => {bar!()};
}

fn main() {
    let res = fo<|>o!();
}
"#,
            expect![[r#"
                foo
            "#]],
        );
    }

    #[test]
    fn macro_expand_with_dollar_crate() {
        check(
            r#"
#[macro_export]
macro_rules! bar {
    () => {0};
}
macro_rules! foo {
    () => {$crate::bar!()};
}

fn main() {
    let res = fo<|>o!();
}
"#,
            expect![[r#"
                foo
                0 "#]],
        );
    }
}
