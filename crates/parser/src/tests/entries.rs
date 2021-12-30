use crate::{LexedStr, PrefixEntryPoint, Step};

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
    // FIXME: this one is wrong
    check_prefix(PrefixEntryPoint::Pat, ".. ..", ".. ..");
}

#[test]
fn ty() {
    check_prefix(PrefixEntryPoint::Ty, "fn() foo", "fn()");
    check_prefix(PrefixEntryPoint::Ty, "Clone + Copy + fn", "Clone + Copy +");
    check_prefix(PrefixEntryPoint::Ty, "struct f", "struct");
}

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
