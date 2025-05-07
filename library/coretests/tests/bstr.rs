use core::bstr::ByteStr;

#[test]
fn test_debug() {
    assert_eq!(
        r#""\0\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x11\x12\r\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f\x80\x81\xfe\xff""#,
        format!("{:?}", ByteStr::new(b"\0\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x11\x12\r\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f\x80\x81\xfe\xff")),
    );
}

#[test]
fn test_display() {
    let b1 = ByteStr::new("abc");
    let b2 = ByteStr::new(b"\xf0\x28\x8c\xbc");

    assert_eq!(&format!("{b1}"), "abc");
    assert_eq!(&format!("{b2}"), "�(��");

    assert_eq!(&format!("{b1:<7}!"), "abc    !");
    assert_eq!(&format!("{b1:>7}!"), "    abc!");
    assert_eq!(&format!("{b1:^7}!"), "  abc  !");
    assert_eq!(&format!("{b1:^6}!"), " abc  !");
    assert_eq!(&format!("{b1:-<7}!"), "abc----!");
    assert_eq!(&format!("{b1:->7}!"), "----abc!");
    assert_eq!(&format!("{b1:-^7}!"), "--abc--!");
    assert_eq!(&format!("{b1:-^6}!"), "-abc--!");

    assert_eq!(&format!("{b2:<7}!"), "�(��   !");
    assert_eq!(&format!("{b2:>7}!"), "   �(��!");
    assert_eq!(&format!("{b2:^7}!"), " �(��  !");
    assert_eq!(&format!("{b2:^6}!"), " �(�� !");
    assert_eq!(&format!("{b2:-<7}!"), "�(��---!");
    assert_eq!(&format!("{b2:->7}!"), "---�(��!");
    assert_eq!(&format!("{b2:-^7}!"), "-�(��--!");
    assert_eq!(&format!("{b2:-^6}!"), "-�(��-!");

    assert_eq!(&format!("{b1:<2}!"), "abc!");
    assert_eq!(&format!("{b1:>2}!"), "abc!");
    assert_eq!(&format!("{b1:^2}!"), "abc!");
    assert_eq!(&format!("{b1:-<2}!"), "abc!");
    assert_eq!(&format!("{b1:->2}!"), "abc!");
    assert_eq!(&format!("{b1:-^2}!"), "abc!");

    assert_eq!(&format!("{b2:<3}!"), "�(��!");
    assert_eq!(&format!("{b2:>3}!"), "�(��!");
    assert_eq!(&format!("{b2:^3}!"), "�(��!");
    assert_eq!(&format!("{b2:^2}!"), "�(��!");
    assert_eq!(&format!("{b2:-<3}!"), "�(��!");
    assert_eq!(&format!("{b2:->3}!"), "�(��!");
    assert_eq!(&format!("{b2:-^3}!"), "�(��!");
    assert_eq!(&format!("{b2:-^2}!"), "�(��!");
}
