use crate::{LexedStr, PrefixEntryPoint, StrStep};

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
    check_prefix(PrefixEntryPoint::Stmt, ";;;", ";");
    check_prefix(PrefixEntryPoint::Stmt, "+", "+");
    check_prefix(PrefixEntryPoint::Stmt, "@", "@");
    check_prefix(PrefixEntryPoint::Stmt, "loop {} - 1", "loop {}");
}

fn check_prefix(entry: PrefixEntryPoint, input: &str, prefix: &str) {
    let lexed = LexedStr::new(input);
    let input = lexed.to_input();
    let output = entry.parse(&input);

    let mut buf = String::new();
    lexed.intersperse_trivia(&output, &mut |step| match step {
        StrStep::Token { kind: _, text } => buf.push_str(text),
        _ => (),
    });
    assert_eq!(buf.trim(), prefix)
}
