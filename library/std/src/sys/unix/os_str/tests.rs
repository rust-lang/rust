use super::*;

#[test]
fn slice_debug_output() {
    let input = Slice::from_u8_slice(b"\xF0hello,\tworld");
    let expected = r#""\xF0hello,\tworld""#;
    let output = format!("{:?}", input);

    assert_eq!(output, expected);
}
