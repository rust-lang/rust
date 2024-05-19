//@ check-pass
// ignore-tidy-tab
//@ edition: 2021

fn main() {
    let s = "\

             ";
    //~^^^ WARNING multiple lines skipped by escaped newline
    assert_eq!(s, "");

    let s = c"foo\
             bar
             ";
    //~^^^ WARNING whitespace symbol '\u{a0}' is not skipped
    assert_eq!(s, c"foo           bar\n             ");

    let s = "a\
 b";
    assert_eq!(s, "ab");

    let s = "a\
	b";
    assert_eq!(s, "ab");

    let s = b"a\
    b";
    //~^^ WARNING whitespace symbol '\u{c}' is not skipped
    // '\x0c' is ASCII whitespace, but it may not need skipped
    // discussion: https://github.com/rust-lang/rust/pull/108403
    assert_eq!(s, b"a\x0cb");
}
