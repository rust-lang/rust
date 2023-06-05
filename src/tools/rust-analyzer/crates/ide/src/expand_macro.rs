use hir::Semantics;
use ide_db::{
    base_db::FileId, helpers::pick_best_token,
    syntax_helpers::insert_whitespace_into_node::insert_ws_into, RootDatabase,
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
// | VS Code | **rust-analyzer: Expand macro recursively**
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

    // due to how rust-analyzer works internally, we need to special case derive attributes,
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

        let name = descended.parent_ancestors().filter_map(ast::Path::cast).last()?.to_string();
        // up map out of the #[derive] expansion
        let token = hir::InFile::new(hir_file, descended).upmap(db)?.value;
        let attr = token.parent_ancestors().find_map(ast::Attr::cast)?;
        let expansions = sema.expand_derive_macro(&attr)?;
        let idx = attr
            .token_tree()?
            .token_trees_and_tokens()
            .filter_map(NodeOrToken::into_token)
            .take_while(|it| it != &token)
            .filter(|it| it.kind() == T![,])
            .count();
        let expansion =
            format(db, SyntaxKind::MACRO_ITEMS, position.file_id, expansions.get(idx).cloned()?);
        Some(ExpandedMacro { name, expansion })
    });

    if derive.is_some() {
        return derive;
    }

    // FIXME: Intermix attribute and bang! expansions
    // currently we only recursively expand one of the two types
    let mut anc = tok.parent_ancestors();
    let (name, expanded, kind) = loop {
        let node = anc.next()?;

        if let Some(item) = ast::Item::cast(node.clone()) {
            if let Some(def) = sema.resolve_attr_macro_call(&item) {
                break (
                    def.name(db).display(db).to_string(),
                    expand_attr_macro_recur(&sema, &item)?,
                    SyntaxKind::MACRO_ITEMS,
                );
            }
        }
        if let Some(mac) = ast::MacroCall::cast(node) {
            let mut name = mac.path()?.segment()?.name_ref()?.to_string();
            name.push('!');
            break (
                name,
                expand_macro_recur(&sema, &mac)?,
                mac.syntax().parent().map(|it| it.kind()).unwrap_or(SyntaxKind::MACRO_ITEMS),
            );
        }
    };

    // FIXME:
    // macro expansion may lose all white space information
    // But we hope someday we can use ra_fmt for that
    let expansion = format(db, kind, position.file_id, expanded);

    Some(ExpandedMacro { name, expansion })
}

fn expand_macro_recur(
    sema: &Semantics<'_, RootDatabase>,
    macro_call: &ast::MacroCall,
) -> Option<SyntaxNode> {
    let expanded = sema.expand(macro_call)?.clone_for_update();
    expand(sema, expanded, ast::MacroCall::cast, expand_macro_recur)
}

fn expand_attr_macro_recur(
    sema: &Semantics<'_, RootDatabase>,
    item: &ast::Item,
) -> Option<SyntaxNode> {
    let expanded = sema.expand_attr_macro(item)?.clone_for_update();
    expand(sema, expanded, ast::Item::cast, expand_attr_macro_recur)
}

fn expand<T: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    expanded: SyntaxNode,
    f: impl FnMut(SyntaxNode) -> Option<T>,
    exp: impl Fn(&Semantics<'_, RootDatabase>, &T) -> Option<SyntaxNode>,
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

fn format(db: &RootDatabase, kind: SyntaxKind, file_id: FileId, expanded: SyntaxNode) -> String {
    let expansion = insert_ws_into(expanded).to_string();

    _format(db, kind, file_id, &expansion).unwrap_or(expansion)
}

#[cfg(any(test, target_arch = "wasm32", target_os = "emscripten"))]
fn _format(
    _db: &RootDatabase,
    _kind: SyntaxKind,
    _file_id: FileId,
    expansion: &str,
) -> Option<String> {
    // remove trailing spaces for test
    use itertools::Itertools;
    Some(expansion.lines().map(|x| x.trim_end()).join("\n"))
}

#[cfg(not(any(test, target_arch = "wasm32", target_os = "emscripten")))]
fn _format(
    db: &RootDatabase,
    kind: SyntaxKind,
    file_id: FileId,
    expansion: &str,
) -> Option<String> {
    use ide_db::base_db::{FileLoader, SourceDatabase};
    // hack until we get hygiene working (same character amount to preserve formatting as much as possible)
    const DOLLAR_CRATE_REPLACE: &str = "__r_a_";
    let expansion = expansion.replace("$crate", DOLLAR_CRATE_REPLACE);
    let (prefix, suffix) = match kind {
        SyntaxKind::MACRO_PAT => ("fn __(", ": u32);"),
        SyntaxKind::MACRO_EXPR | SyntaxKind::MACRO_STMTS => ("fn __() {", "}"),
        SyntaxKind::MACRO_TYPE => ("type __ =", ";"),
        _ => ("", ""),
    };
    let expansion = format!("{prefix}{expansion}{suffix}");

    let &crate_id = db.relevant_crates(file_id).iter().next()?;
    let edition = db.crate_graph()[crate_id].edition;

    let mut cmd = std::process::Command::new(toolchain::rustfmt());
    cmd.arg("--edition");
    cmd.arg(edition.to_string());

    let mut rustfmt = cmd
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;

    std::io::Write::write_all(&mut rustfmt.stdin.as_mut()?, expansion.as_bytes()).ok()?;

    let output = rustfmt.wait_with_output().ok()?;
    let captured_stdout = String::from_utf8(output.stdout).ok()?;

    if output.status.success() && !captured_stdout.trim().is_empty() {
        let output = captured_stdout.replace(DOLLAR_CRATE_REPLACE, "$crate");
        let output = output.trim().strip_prefix(prefix)?;
        let output = match kind {
            SyntaxKind::MACRO_PAT => {
                output.strip_suffix(suffix).or_else(|| output.strip_suffix(": u32,\n);"))?
            }
            _ => output.strip_suffix(suffix)?,
        };
        let trim_indent = stdx::trim_indent(output);
        tracing::debug!("expand_macro: formatting succeeded");
        Some(trim_indent)
    } else {
        None
    }
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
                bar!
                5i64 as _"#]],
        );
    }

    #[test]
    fn macro_expand_underscore() {
        check(
            r#"
macro_rules! bar {
    ($i:tt) => { for _ in 0..$i {} }
}
fn main() {
    ba$0r!(42);
}
"#,
            expect![[r#"
                bar!
                for _ in 0..42{}"#]],
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
                foo!
                fn b(){}"#]],
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
                foo!
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
                match_ast!
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
                match_ast!
                {}"#]],
        );
    }

    #[test]
    fn macro_expand_inner_macro_rules() {
        check(
            r#"
macro_rules! foo {
    ($t:tt) => {{
        macro_rules! bar {
            () => {
                $t
            }
        }
        bar!()
    }};
}

fn main() {
    foo$0!(42);
}
            "#,
            expect![[r#"
                foo!
                {
                  macro_rules! bar {
                    () => {
                      42
                    }
                  }
                  42
                }"#]],
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
                foo!
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
                foo!
                0"#]],
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
                foo!
                fn f<T>(_: &dyn ::std::marker::Copy){}"#]],
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
                impl < >core::clone::Clone for Foo< >where {
                  fn clone(&self) -> Self {
                    match self {
                      Foo{}
                       => Foo{}
                      ,

                      }
                  }

                  }"#]],
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
                impl < >core::marker::Copy for Foo< >where{}"#]],
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
                impl < >core::marker::Copy for Foo< >where{}"#]],
        );
        check(
            r#"
//- minicore: copy, clone, derive

#[derive(Copy, Cl$0one)]
struct Foo {}
"#,
            expect![[r#"
                Clone
                impl < >core::clone::Clone for Foo< >where {
                  fn clone(&self) -> Self {
                    match self {
                      Foo{}
                       => Foo{}
                      ,

                      }
                  }

                  }"#]],
        );
    }
}
