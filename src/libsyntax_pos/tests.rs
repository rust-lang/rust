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
