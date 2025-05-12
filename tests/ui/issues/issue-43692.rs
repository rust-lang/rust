//@ run-pass
fn main() {
    assert_eq!('\u{10__FFFF}', '\u{10FFFF}');
    assert_eq!("\u{10_F0FF__}foo\u{1_0_0_0__}", "\u{10F0FF}foo\u{1000}");
}
