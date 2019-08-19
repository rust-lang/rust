use super::*;

#[test]
fn test_lookup_line() {

    let lines = &[BytePos(3), BytePos(17), BytePos(28)];

    assert_eq!(lookup_line(lines, BytePos(0)), -1);
    assert_eq!(lookup_line(lines, BytePos(3)),  0);
    assert_eq!(lookup_line(lines, BytePos(4)),  0);

    assert_eq!(lookup_line(lines, BytePos(16)), 0);
    assert_eq!(lookup_line(lines, BytePos(17)), 1);
    assert_eq!(lookup_line(lines, BytePos(18)), 1);

    assert_eq!(lookup_line(lines, BytePos(28)), 2);
    assert_eq!(lookup_line(lines, BytePos(29)), 2);
}

#[test]
fn test_normalize_newlines() {
    fn check(before: &str, after: &str) {
        let mut actual = before.to_string();
        normalize_newlines(&mut actual);
        assert_eq!(actual.as_str(), after);
    }
    check("", "");
    check("\n", "\n");
    check("\r", "\r");
    check("\r\r", "\r\r");
    check("\r\n", "\n");
    check("hello world", "hello world");
    check("hello\nworld", "hello\nworld");
    check("hello\r\nworld", "hello\nworld");
    check("\r\nhello\r\nworld\r\n", "\nhello\nworld\n");
    check("\r\r\n", "\r\n");
    check("hello\rworld", "hello\rworld");
}
