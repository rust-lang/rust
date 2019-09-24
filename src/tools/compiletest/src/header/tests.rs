use super::*;

#[test]
fn test_parse_normalization_string() {
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits)\".";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits)\".");

    // Nothing to normalize (No quotes)
    let mut s = "normalize-stderr-32bit: something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, r#"normalize-stderr-32bit: something (32 bits) -> something ($WORD bits)."#);

    // Nothing to normalize (Only a single quote)
    let mut s = "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).");

    // Nothing to normalize (Three quotes)
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits).");
}
