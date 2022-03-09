use hir::Semantics;
use ide_db::{
    helpers::pick_best_token, syntax_helpers::insert_whitespace_into_node::insert_ws_into,
    RootDatabase,
};
use syntax::{ast, ted, AstNode, NodeOrToken, SyntaxKind, SyntaxNode, T};

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
//
// image::https://user-images.githubusercontent.com/48062697/113020648-b3973180-917a-11eb-84a9-ecb921293dc5.gif[]
pub(crate) fn expand_macro(db: &RootDatabase, position: FilePosition) -> Option<ExpandedMacro> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id);

    let tok = pick_best_token(file.syntax().token_at_offset(position.offset), |kind| match kind {
        SyntaxKind::IDENT => 1,
        _ => 0,
    })?;

    // due to how Rust Analyzer works internally, we need to special case derive attributes,
    // otherwise they might not get found, e.g. here with the cursor at $0 `#[attr]` would expand:
    // ```
    // #[attr]
    // #[derive($0Foo)]
    // struct Bar;
    // ```

    let derive = sema.descend_into_macros(tok.clone()).into_iter().find_map(|descended| {
        let hir_file = sema.hir_file_for(&descended.parent()?);
        if !hir_file.is_derive_attr_pseudo_expansion(db) {
            return None;
        }

        let name = descended.ancestors().filter_map(ast::Path::cast).last()?.to_string();
        // up map out of the #[derive] expansion
        let token = hir::InFile::new(hir_file, descended).upmap(db)?.value;
        let attr = token.ancestors().find_map(ast::Attr::cast)?;
        let expansions = sema.expand_derive_macro(&attr)?;
        let idx = attr
            .token_tree()?
            .token_trees_and_tokens()
            .filter_map(NodeOrToken::into_token)
            .take_while(|it| it != &token)
            .filter(|it| it.kind() == T![,])
            .count();
        Some(ExpandedMacro {
            name,
            expansion: expansions.get(idx).cloned().map(insert_ws_into)?.to_string(),
        })
    });

    if derive.is_some() {
        return derive;
    }

    // FIXME: Intermix attribute and bang! expansions
    // currently we only recursively expand one of the two types
    let mut expanded = None;
    let mut name = None;
    for node in tok.ancestors() {
        if let Some(item) = ast::Item::cast(node.clone()) {
            if let Some(def) = sema.resolve_attr_macro_call(&item) {
                name = Some(def.name(db).to_string());
                expanded = expand_attr_macro_recur(&sema, &item);
                break;
            }
        }
        if let Some(mac) = ast::MacroCall::cast(node) {
            name = Some(mac.path()?.segment()?.name_ref()?.to_string());
            expanded = expand_macro_recur(&sema, &mac);
            break;
        }
    }

    // FIXME:
    // macro expansion may lose all white space information
    // But we hope someday we can use ra_fmt for that
    let expansion = insert_ws_into(expanded?).to_string();
    Some(ExpandedMacro { name: name.unwrap_or_else(|| "???".to_owned()), expansion })
}

fn expand_macro_recur(
    sema: &Semantics<RootDatabase>,
    macro_call: &ast::MacroCall,
) -> Option<SyntaxNode> {
    let expanded = sema.expand(macro_call)?.clone_for_update();
    expand(sema, expanded, ast::MacroCall::cast, expand_macro_recur)
}

fn expand_attr_macro_recur(sema: &Semantics<RootDatabase>, item: &ast::Item) -> Option<SyntaxNode> {
    let expanded = sema.expand_attr_macro(item)?.clone_for_update();
    expand(sema, expanded, ast::Item::cast, expand_attr_macro_recur)
}

fn expand<T: AstNode>(
    sema: &Semantics<RootDatabase>,
    expanded: SyntaxNode,
    f: impl FnMut(SyntaxNode) -> Option<T>,
    exp: impl Fn(&Semantics<RootDatabase>, &T) -> Option<SyntaxNode>,
) -> Option<SyntaxNode> {
    let children = expanded.descendants().filter_map(f);
    let mut replacements = Vec::new();

    for child in children {
        if let Some(new_node) = exp(sema, &child) {
            // check if the whole original syntax is replaced
            if expanded == *child.syntax() {
                return Some(new_node);
            }
            replacements.push((child, new_node));
        }
    }

    replacements.into_iter().rev().for_each(|(old, new)| ted::replace(old.syntax(), new));
    Some(expanded)
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::fixture;

    #[track_caller]
    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, pos) = fixture::position(ra_fixture);
        let expansion = analysis.expand_macro(pos).unwrap().unwrap();
        let actual = format!("{}\n{}", expansion.name, expansion.expansion);
        expect.assert_eq(&actual);
    }

    #[test]
    fn macro_expand_as_keyword() {
        check(
            r#"
macro_rules! bar {
    ($i:tt) => { $i as _ }
}
fn main() {
    let x: u64 = ba$0r!(5i64);
}
"#,
            expect![[r#"
                bar
                5i64 as _"#]],
        );
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
f$0oo!();
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
f$0oo!();
        "#,
            expect![[r#"
                foo
                fn some_thing() -> u32 {
                  let a = 0;
                  a+10
                }
            "#]],
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
    mat$0ch_ast! {
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
        let res = mat$0ch_ast! { match c {}};
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
    let res = fo$0o!();
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
    let res = fo$0o!();
}
"#,
            expect![[r#"
                foo
                0 "#]],
        );
    }

    #[test]
    fn macro_expand_with_dyn_absolute_path() {
        check(
            r#"
macro_rules! foo {
    () => {fn f<T>(_: &dyn ::std::marker::Copy) {}};
}

fn main() {
    let res = fo$0o!();
}
"#,
            expect![[r#"
                foo
                fn f<T>(_: &dyn ::std::marker::Copy){}
            "#]],
        );
    }

    #[test]
    fn macro_expand_derive() {
        check(
            r#"
//- proc_macros: identity
//- minicore: clone, derive

#[proc_macros::identity]
#[derive(C$0lone)]
struct Foo {}
"#,
            expect![[r#"
                Clone
                impl < >core::clone::Clone for Foo< >{}

            "#]],
        );
    }

    #[test]
    fn macro_expand_derive2() {
        check(
            r#"
//- minicore: copy, clone, derive

#[derive(Cop$0y)]
#[derive(Clone)]
struct Foo {}
"#,
            expect![[r#"
                Copy
                impl < >core::marker::Copy for Foo< >{}

            "#]],
        );
    }

    #[test]
    fn macro_expand_derive_multi() {
        check(
            r#"
//- minicore: copy, clone, derive

#[derive(Cop$0y, Clone)]
struct Foo {}
"#,
            expect![[r#"
                Copy
                impl < >core::marker::Copy for Foo< >{}

            "#]],
        );
        check(
            r#"
//- minicore: copy, clone, derive

#[derive(Copy, Cl$0one)]
struct Foo {}
"#,
            expect![[r#"
                Clone
                impl < >core::clone::Clone for Foo< >{}

            "#]],
        );
    }
}
