use super::*;

#[test]
fn test_lookup_line() {
    let source = "abcdefghijklm\nabcdefghij\n...".to_owned();
    let mut sf =
        SourceFile::new(FileName::Anon(Hash64::ZERO), source, SourceFileHashAlgorithm::Sha256)
            .unwrap();
    sf.start_pos = BytePos(3);
    assert_eq!(sf.lines(), &[RelativeBytePos(0), RelativeBytePos(14), RelativeBytePos(25)]);

    assert_eq!(sf.lookup_line(RelativeBytePos(0)), Some(0));
    assert_eq!(sf.lookup_line(RelativeBytePos(1)), Some(0));

    assert_eq!(sf.lookup_line(RelativeBytePos(13)), Some(0));
    assert_eq!(sf.lookup_line(RelativeBytePos(14)), Some(1));
    assert_eq!(sf.lookup_line(RelativeBytePos(15)), Some(1));

    assert_eq!(sf.lookup_line(RelativeBytePos(25)), Some(2));
    assert_eq!(sf.lookup_line(RelativeBytePos(26)), Some(2));
}

#[test]
fn test_normalize_newlines() {
    fn check(before: &str, after: &str, expected_positions: &[u32]) {
        let mut actual = before.to_string();
        let mut actual_positions = vec![];
        normalize_newlines(&mut actual, &mut actual_positions);
        let actual_positions: Vec<_> = actual_positions.into_iter().map(|nc| nc.pos.0).collect();
        assert_eq!(actual.as_str(), after);
        assert_eq!(actual_positions, expected_positions);
    }
    check("", "", &[]);
    check("\n", "\n", &[]);
    check("\r", "\r", &[]);
    check("\r\r", "\r\r", &[]);
    check("\r\n", "\n", &[1]);
    check("hello world", "hello world", &[]);
    check("hello\nworld", "hello\nworld", &[]);
    check("hello\r\nworld", "hello\nworld", &[6]);
    check("\r\nhello\r\nworld\r\n", "\nhello\nworld\n", &[1, 7, 13]);
    check("\r\r\n", "\r\n", &[2]);
    check("hello\rworld", "hello\rworld", &[]);
}
