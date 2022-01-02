use expect_test::expect;

use crate::{LexedStr, PrefixEntryPoint, Step, TopEntryPoint};

#[test]
fn vis() {
    check_prefix(PrefixEntryPoint::Vis, "pub(crate) fn foo() {}", "pub(crate)");
    check_prefix(PrefixEntryPoint::Vis, "fn foo() {}", "");
    check_prefix(PrefixEntryPoint::Vis, "pub(fn foo() {}", "pub");
    check_prefix(PrefixEntryPoint::Vis, "pub(crate fn foo() {}", "pub(crate");
    check_prefix(PrefixEntryPoint::Vis, "crate fn foo() {}", "crate");
}

#[test]
fn block() {
    check_prefix(PrefixEntryPoint::Block, "{}, 92", "{}");
    check_prefix(PrefixEntryPoint::Block, "{, 92)", "{, 92)");
    check_prefix(PrefixEntryPoint::Block, "()", "");
}

#[test]
fn stmt() {
    check_prefix(PrefixEntryPoint::Stmt, "92; fn", "92");
    check_prefix(PrefixEntryPoint::Stmt, "let _ = 92; 1", "let _ = 92");
    check_prefix(PrefixEntryPoint::Stmt, "pub fn f() {} = 92", "pub fn f() {}");
    check_prefix(PrefixEntryPoint::Stmt, "struct S;;", "struct S;");
    check_prefix(PrefixEntryPoint::Stmt, "fn f() {};", "fn f() {}");
    check_prefix(PrefixEntryPoint::Stmt, ";;;", ";");
    check_prefix(PrefixEntryPoint::Stmt, "+", "+");
    check_prefix(PrefixEntryPoint::Stmt, "@", "@");
    check_prefix(PrefixEntryPoint::Stmt, "loop {} - 1", "loop {}");
}

#[test]
fn pat() {
    check_prefix(PrefixEntryPoint::Pat, "x y", "x");
    check_prefix(PrefixEntryPoint::Pat, "fn f() {}", "fn");
    // FIXME: This one is wrong, we should consume only one pattern.
    check_prefix(PrefixEntryPoint::Pat, ".. ..", ".. ..");
}

#[test]
fn ty() {
    check_prefix(PrefixEntryPoint::Ty, "fn() foo", "fn()");
    check_prefix(PrefixEntryPoint::Ty, "Clone + Copy + fn", "Clone + Copy +");
    check_prefix(PrefixEntryPoint::Ty, "struct f", "struct");
}

#[test]
fn expr() {
    check_prefix(PrefixEntryPoint::Expr, "92 92", "92");
    check_prefix(PrefixEntryPoint::Expr, "+1", "+");
    check_prefix(PrefixEntryPoint::Expr, "-1", "-1");
    check_prefix(PrefixEntryPoint::Expr, "fn foo() {}", "fn");
    check_prefix(PrefixEntryPoint::Expr, "#[attr] ()", "#[attr] ()");
}

#[test]
fn path() {
    check_prefix(PrefixEntryPoint::Path, "foo::bar baz", "foo::bar");
    check_prefix(PrefixEntryPoint::Path, "foo::<> baz", "foo::<>");
    check_prefix(PrefixEntryPoint::Path, "foo<> baz", "foo<>");
    check_prefix(PrefixEntryPoint::Path, "Fn() -> i32?", "Fn() -> i32");
    // FIXME: This shouldn't be accepted as path actually.
    check_prefix(PrefixEntryPoint::Path, "<_>::foo", "<_>::foo");
}

#[test]
fn item() {
    // FIXME: This shouldn't consume the semicolon.
    check_prefix(PrefixEntryPoint::Item, "fn foo() {};", "fn foo() {};");
    check_prefix(PrefixEntryPoint::Item, "#[attr] pub struct S {} 92", "#[attr] pub struct S {}");
    check_prefix(PrefixEntryPoint::Item, "item!{}?", "item!{}");
    check_prefix(PrefixEntryPoint::Item, "????", "?");
}

#[test]
fn meta_item() {
    check_prefix(PrefixEntryPoint::MetaItem, "attr, ", "attr");
    check_prefix(
        PrefixEntryPoint::MetaItem,
        "attr(some token {stream});",
        "attr(some token {stream})",
    );
    check_prefix(PrefixEntryPoint::MetaItem, "path::attr = 2 * 2!", "path::attr = 2 * 2");
}

#[track_caller]
fn check_prefix(entry: PrefixEntryPoint, input: &str, prefix: &str) {
    let lexed = LexedStr::new(input);
    let input = lexed.to_input();

    let mut n_tokens = 0;
    for step in entry.parse(&input).iter() {
        match step {
            Step::Token { n_input_tokens, .. } => n_tokens += n_input_tokens as usize,
            Step::Enter { .. } | Step::Exit | Step::Error { .. } => (),
        }
    }

    let mut i = 0;
    loop {
        if n_tokens == 0 {
            break;
        }
        if !lexed.kind(i).is_trivia() {
            n_tokens -= 1;
        }
        i += 1;
    }
    let buf = &lexed.as_str()[..lexed.text_start(i)];
    assert_eq!(buf, prefix);
}

#[test]
fn source_file() {
    check_top(
        TopEntryPoint::SourceFile,
        "",
        expect![[r#"
        SOURCE_FILE
    "#]],
    );

    check_top(
        TopEntryPoint::SourceFile,
        "struct S;",
        expect![[r#"
        SOURCE_FILE
          STRUCT
            STRUCT_KW "struct"
            WHITESPACE " "
            NAME
              IDENT "S"
            SEMICOLON ";"
    "#]],
    );

    check_top(
        TopEntryPoint::SourceFile,
        "@error@",
        expect![[r#"
        SOURCE_FILE
          ERROR
            AT "@"
          MACRO_CALL
            PATH
              PATH_SEGMENT
                NAME_REF
                  IDENT "error"
          ERROR
            AT "@"
        error 0: expected an item
        error 6: expected BANG
        error 6: expected `{`, `[`, `(`
        error 6: expected SEMICOLON
        error 6: expected an item
    "#]],
    );
}

#[track_caller]
fn check_top(entry: TopEntryPoint, input: &str, expect: expect_test::Expect) {
    let (parsed, _errors) = super::parse(entry, input);
    expect.assert_eq(&parsed)
}
