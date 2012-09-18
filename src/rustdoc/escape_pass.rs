//! Escapes text sequences

export mk_pass;

fn mk_pass() -> Pass {
    text_pass::mk_pass(~"escape", escape)
}

fn escape(s: ~str) -> ~str {
    str::replace(s, ~"\\", ~"\\\\")
}

#[test]
fn should_escape_backslashes() {
    let s = ~"\\n";
    let r = escape(s);
    assert r == ~"\\\\n";
}
