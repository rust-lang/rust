use hir::db::ExpandDatabase;
use hir::{InFile, MacroFileIdExt, Semantics};
use ide_db::base_db::CrateId;
use ide_db::{
    helpers::pick_best_token, syntax_helpers::prettify_macro_expansion, FileId, RootDatabase,
};
use span::{Edition, SpanMap, SyntaxContextId, TextRange, TextSize};
use syntax::{ast, ted, AstNode, NodeOrToken, SyntaxKind, SyntaxNode, T};

use crate::FilePosition;

pub struct ExpandedMacro {
    pub name: String,
    pub expansion: String,
}

// Feature: Expand Macro Recursively
//
// Shows the full macro expansion of the macro at the current caret position.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Expand macro recursively at caret**
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113020648-b3973180-917a-11eb-84a9-ecb921293dc5.gif[]
pub(crate) fn expand_macro(db: &RootDatabase, position: FilePosition) -> Option<ExpandedMacro> {
    let sema = Semantics::new(db);
    let file = sema.parse_guess_edition(position.file_id);
    let krate = sema.file_to_module_def(position.file_id)?.krate().into();

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

    let derive = sema.descend_into_macros_exact(tok.clone()).into_iter().find_map(|descended| {
        let macro_file = sema.hir_file_for(&descended.parent()?).macro_file()?;
        if !macro_file.is_derive_attr_pseudo_expansion(db) {
            return None;
        }

        let name = descended.parent_ancestors().filter_map(ast::Path::cast).last()?.to_string();
        // up map out of the #[derive] expansion
        let InFile { file_id, value: tokens } =
            hir::InMacroFile::new(macro_file, descended).upmap_once(db);
        let token = sema.parse_or_expand(file_id).covering_element(tokens[0]).into_token()?;
        let attr = token.parent_ancestors().find_map(ast::Attr::cast)?;
        let expansions = sema.expand_derive_macro(&attr)?;
        let idx = attr
            .token_tree()?
            .token_trees_and_tokens()
            .filter_map(NodeOrToken::into_token)
            .take_while(|it| it != &token)
            .filter(|it| it.kind() == T![,])
            .count();
        let expansion = expansions.get(idx)?.clone();
        let expansion_file_id = sema.hir_file_for(&expansion).macro_file()?;
        let expansion_span_map = db.expansion_span_map(expansion_file_id);
        let expansion = format(
            db,
            SyntaxKind::MACRO_ITEMS,
            position.file_id,
            expansion,
            &expansion_span_map,
            krate,
        );
        Some(ExpandedMacro { name, expansion })
    });

    if derive.is_some() {
        return derive;
    }

    let mut anc = tok.parent_ancestors();
    let mut span_map = SpanMap::empty();
    let (name, expanded, kind) = loop {
        let node = anc.next()?;

        if let Some(item) = ast::Item::cast(node.clone()) {
            if let Some(def) = sema.resolve_attr_macro_call(&item) {
                break (
                    def.name(db)
                        .display(
                            db,
                            sema.attach_first_edition(position.file_id)
                                .map(|it| it.edition())
                                .unwrap_or(Edition::CURRENT),
                        )
                        .to_string(),
                    expand_macro_recur(&sema, &item, &mut span_map, TextSize::new(0))?,
                    SyntaxKind::MACRO_ITEMS,
                );
            }
        }
        if let Some(mac) = ast::MacroCall::cast(node) {
            let mut name = mac.path()?.segment()?.name_ref()?.to_string();
            name.push('!');
            let syntax_kind =
                mac.syntax().parent().map(|it| it.kind()).unwrap_or(SyntaxKind::MACRO_ITEMS);
            break (
                name,
                expand_macro_recur(
                    &sema,
                    &ast::Item::MacroCall(mac),
                    &mut span_map,
                    TextSize::new(0),
                )?,
                syntax_kind,
            );
        }
    };

    // FIXME:
    // macro expansion may lose all white space information
    // But we hope someday we can use ra_fmt for that
    let expansion = format(db, kind, position.file_id, expanded, &span_map, krate);

    Some(ExpandedMacro { name, expansion })
}

fn expand_macro_recur(
    sema: &Semantics<'_, RootDatabase>,
    macro_call: &ast::Item,
    result_span_map: &mut SpanMap<SyntaxContextId>,
    offset_in_original_node: TextSize,
) -> Option<SyntaxNode> {
    let expanded = match macro_call {
        item @ ast::Item::MacroCall(macro_call) => sema
            .expand_attr_macro(item)
            .or_else(|| sema.expand_allowed_builtins(macro_call))?
            .clone_for_update(),
        item => sema.expand_attr_macro(item)?.clone_for_update(),
    };
    let file_id =
        sema.hir_file_for(&expanded).macro_file().expect("expansion must produce a macro file");
    let expansion_span_map = sema.db.expansion_span_map(file_id);
    result_span_map.merge(
        TextRange::at(offset_in_original_node, macro_call.syntax().text_range().len()),
        expanded.text_range().len(),
        &expansion_span_map,
    );
    Some(expand(sema, expanded, result_span_map, u32::from(offset_in_original_node) as i32))
}

fn expand(
    sema: &Semantics<'_, RootDatabase>,
    expanded: SyntaxNode,
    result_span_map: &mut SpanMap<SyntaxContextId>,
    mut offset_in_original_node: i32,
) -> SyntaxNode {
    let children = expanded.descendants().filter_map(ast::Item::cast);
    let mut replacements = Vec::new();

    for child in children {
        if let Some(new_node) = expand_macro_recur(
            sema,
            &child,
            result_span_map,
            TextSize::new(
                (offset_in_original_node + (u32::from(child.syntax().text_range().start()) as i32))
                    as u32,
            ),
        ) {
            offset_in_original_node = offset_in_original_node
                + (u32::from(new_node.text_range().len()) as i32)
                - (u32::from(child.syntax().text_range().len()) as i32);
            // check if the whole original syntax is replaced
            if expanded == *child.syntax() {
                return new_node;
            }
            replacements.push((child, new_node));
        }
    }

    replacements.into_iter().rev().for_each(|(old, new)| ted::replace(old.syntax(), new));
    expanded
}

fn format(
    db: &RootDatabase,
    kind: SyntaxKind,
    file_id: FileId,
    expanded: SyntaxNode,
    span_map: &SpanMap<SyntaxContextId>,
    krate: CrateId,
) -> String {
    let expansion = prettify_macro_expansion(db, expanded, span_map, krate).to_string();

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
    const BUILTIN_REPLACE: &str = "builtin__POUND";
    let expansion =
        expansion.replace("$crate", DOLLAR_CRATE_REPLACE).replace("builtin #", BUILTIN_REPLACE);
    let (prefix, suffix) = match kind {
        SyntaxKind::MACRO_PAT => ("fn __(", ": u32);"),
        SyntaxKind::MACRO_EXPR | SyntaxKind::MACRO_STMTS => ("fn __() {", "}"),
        SyntaxKind::MACRO_TYPE => ("type __ =", ";"),
        _ => ("", ""),
    };
    let expansion = format!("{prefix}{expansion}{suffix}");

    let &crate_id = db.relevant_crates(file_id).iter().next()?;
    let edition = db.crate_graph()[crate_id].edition;

    let mut cmd = std::process::Command::new(toolchain::Tool::Rustfmt.path());
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
        let output = captured_stdout
            .replace(DOLLAR_CRATE_REPLACE, "$crate")
            .replace(BUILTIN_REPLACE, "builtin #");
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
    fn expand_allowed_builtin_macro() {
        check(
            r#"
//- minicore: concat
$0concat!("test", 10, 'b', true);"#,
            expect![[r#"
                concat!
                "test10btrue""#]],
        );
    }

    #[test]
    fn do_not_expand_disallowed_macro() {
        let (analysis, pos) = fixture::position(
            r#"
//- minicore: asm
$0asm!("0x300, x0");"#,
        );
        let expansion = analysis.expand_macro(pos).unwrap();
        assert!(expansion.is_none());
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

    #[test]
    fn dollar_crate() {
        check(
            r#"
//- /a.rs crate:a
pub struct Foo;
#[macro_export]
macro_rules! m {
    ( $i:ident ) => { $crate::Foo; $crate::Foo; $i::Foo; };
}
//- /b.rs crate:b deps:a
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { a::m!($crate); $crate::Foo; $crate::Foo; };
}
//- /c.rs crate:c deps:b,a
pub struct Foo;
#[macro_export]
macro_rules! m {
    () => { b::m!(); $crate::Foo; $crate::Foo; };
}
fn bar() {
    m$0!();
}
"#,
            expect![[r#"
m!
a::Foo;
a::Foo;
b::Foo;
;
b::Foo;
b::Foo;
;
crate::Foo;
crate::Foo;"#]],
        );
    }
}
