use super::unescape_llvm_string_contents;

// Tests for `unescape_llvm_string_contents`:

#[test]
fn unescape_empty() {
    assert_eq!(unescape_llvm_string_contents(""), &[]);
}

#[test]
fn unescape_noop() {
    let input = "The quick brown fox jumps over the lazy dog.";
    assert_eq!(unescape_llvm_string_contents(input), input.as_bytes());
}

#[test]
fn unescape_backslash() {
    let input = r"\\Hello\\world\\";
    assert_eq!(unescape_llvm_string_contents(input), r"\Hello\world\".as_bytes());
}

#[test]
fn unescape_hex() {
    let input = r"\01\02\03\04\0a\0b\0C\0D\fd\fE\FF";
    let expected: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x0a, 0x0b, 0x0c, 0x0d, 0xfd, 0xfe, 0xff];
    assert_eq!(unescape_llvm_string_contents(input), expected);
}

#[test]
fn unescape_mixed() {
    let input = r"\\01.\5c\5c";
    let expected: &[u8] = br"\01.\\";
    assert_eq!(unescape_llvm_string_contents(input), expected);
}
