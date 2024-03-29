use crate::{LexedStr, PrefixEntryPoint, Step};

#[test]
fn vis() {
    check(PrefixEntryPoint::Vis, "pub(crate) fn foo() {}", "pub(crate)");
    check(PrefixEntryPoint::Vis, "fn foo() {}", "");
    check(PrefixEntryPoint::Vis, "pub(fn foo() {}", "pub");
    check(PrefixEntryPoint::Vis, "pub(crate fn foo() {}", "pub(crate");
}

#[test]
fn block() {
    check(PrefixEntryPoint::Block, "{}, 92", "{}");
    check(PrefixEntryPoint::Block, "{, 92)", "{, 92)");
    check(PrefixEntryPoint::Block, "()", "");
}

#[test]
fn stmt() {
    check(PrefixEntryPoint::Stmt, "92; fn", "92");
    check(PrefixEntryPoint::Stmt, "let _ = 92; 1", "let _ = 92");
    check(PrefixEntryPoint::Stmt, "pub fn f() {} = 92", "pub fn f() {}");
    check(PrefixEntryPoint::Stmt, "struct S;;", "struct S;");
    check(PrefixEntryPoint::Stmt, "fn f() {};", "fn f() {}");
    check(PrefixEntryPoint::Stmt, ";;;", ";");
    check(PrefixEntryPoint::Stmt, "+", "+");
    check(PrefixEntryPoint::Stmt, "@", "@");
    check(PrefixEntryPoint::Stmt, "loop {} - 1", "loop {}");
}

#[test]
fn pat() {
    check(PrefixEntryPoint::Pat, "x y", "x");
    check(PrefixEntryPoint::Pat, "fn f() {}", "fn");
    check(PrefixEntryPoint::Pat, ".. ..", "..");
}

#[test]
fn ty() {
    check(PrefixEntryPoint::Ty, "fn() foo", "fn()");
    check(PrefixEntryPoint::Ty, "Clone + Copy + fn", "Clone + Copy +");
    check(PrefixEntryPoint::Ty, "struct f", "struct");
}

#[test]
fn expr() {
    check(PrefixEntryPoint::Expr, "92 92", "92");
    check(PrefixEntryPoint::Expr, "+1", "+");
    check(PrefixEntryPoint::Expr, "-1", "-1");
    check(PrefixEntryPoint::Expr, "fn foo() {}", "fn");
    check(PrefixEntryPoint::Expr, "#[attr] ()", "#[attr] ()");
    check(PrefixEntryPoint::Expr, "foo.0", "foo.0");
    check(PrefixEntryPoint::Expr, "foo.0.1", "foo.0.1");
    check(PrefixEntryPoint::Expr, "foo.0. foo", "foo.0. foo");
}

#[test]
fn path() {
    check(PrefixEntryPoint::Path, "foo::bar baz", "foo::bar");
    check(PrefixEntryPoint::Path, "foo::<> baz", "foo::<>");
    check(PrefixEntryPoint::Path, "foo<> baz", "foo<>");
    check(PrefixEntryPoint::Path, "Fn() -> i32?", "Fn() -> i32");
    // FIXME: This shouldn't be accepted as path actually.
    check(PrefixEntryPoint::Path, "<_>::foo", "<_>::foo");
}

#[test]
fn item() {
    // FIXME: This shouldn't consume the semicolon.
    check(PrefixEntryPoint::Item, "fn foo() {};", "fn foo() {};");
    check(PrefixEntryPoint::Item, "#[attr] pub struct S {} 92", "#[attr] pub struct S {}");
    check(PrefixEntryPoint::Item, "item!{}?", "item!{}");
    check(PrefixEntryPoint::Item, "????", "?");
}

#[test]
fn meta_item() {
    check(PrefixEntryPoint::MetaItem, "attr, ", "attr");
    check(PrefixEntryPoint::MetaItem, "attr(some token {stream});", "attr(some token {stream})");
    check(PrefixEntryPoint::MetaItem, "path::attr = 2 * 2!", "path::attr = 2 * 2");
}

#[track_caller]
fn check(entry: PrefixEntryPoint, input: &str, prefix: &str) {
    let lexed = LexedStr::new(input);
    let input = lexed.to_input();

    let mut n_tokens = 0;
    for step in entry.parse(&input).iter() {
        match step {
            Step::Token { n_input_tokens, .. } => n_tokens += n_input_tokens as usize,
            Step::FloatSplit { .. } => n_tokens += 1,
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
