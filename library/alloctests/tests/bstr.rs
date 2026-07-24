use alloc::bstr::ByteString;
use core::assert_matches;

#[test]
fn test_debug() {
    let b1 = ByteString(
        b"\0\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x11\x12\r\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f\x80\x81\xfe\xff".to_vec()
    );
    assert_eq!(
        r#""\0\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x11\x12\r\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f\x80\x81\xfe\xff""#,
        format!("{:?}", b1),
    );
}

#[test]
fn test_display() {
    let b1 = ByteString(b"abc".to_vec());
    let b2 = ByteString(b"\xf0\x28\x8c\xbc".to_vec());

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

    assert_eq!(&format!("{b1:.1}!"), &format!("{:.1}!", "abc"));
    assert_eq!(&format!("{b1:.2}!"), &format!("{:.2}!", "abc"));
    assert_eq!(&format!("{b1:.3}!"), &format!("{:.3}!", "abc"));
    assert_eq!(&format!("{b1:-<5.2}!"), &format!("{:-<5.2}!", "abc"));
    assert_eq!(&format!("{b1:-^5.2}!"), &format!("{:-^5.2}!", "abc"));
    assert_eq!(&format!("{b1:->5.2}!"), &format!("{:->5.2}!", "abc"));

    assert_eq!(&format!("{b2:.1}!"), "�!");
    assert_eq!(&format!("{b2:.2}!"), "�(!");
    assert_eq!(&format!("{b2:.3}!"), "�(�!");
    assert_eq!(&format!("{b2:.4}!"), "�(��!");
    assert_eq!(&format!("{b2:-<6.3}!"), "�(�---!");
    assert_eq!(&format!("{b2:-^6.3}!"), "-�(�--!");
    assert_eq!(&format!("{b2:->6.3}!"), "---�(�!");
}

#[test]
fn test_to_string() {
    let b1 = ByteString(b"abc".to_vec());
    let b2 = ByteString(b"\xf0\x28\x8c\xbc".to_vec());

    assert_eq!(Ok("abc".to_string()), b1.to_string());
    assert_matches!(b2.to_string(), Err(std::string::FromUtf8Error { .. }));

    // Can still directly use the trait
    assert_eq!("abc".to_string(), ToString::to_string(b1));
    assert_eq!("�(��".to_string(), ToString::to_string(b2));
}
