#[test]
fn test_char_add() {
    let a = char::from("🎈");
    let b = char::from('🎉');
    let c = a + b;

    assert_eq!(c, "🎈🎉");
}
