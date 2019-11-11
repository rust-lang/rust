#[test]
fn test_char_add() {
    let a = '🎈';
    let b = '🎉';
    let c = a + b;

    assert_eq!(c, "🎈🎉");
}
