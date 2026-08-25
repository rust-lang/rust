use {serde_test::*, std::ops, text_size::*};

fn size(x: u32) -> TextSize {
    TextSize::from(x)
}

fn range(x: ops::Range<u32>) -> TextRange {
    TextRange::new(x.start.into(), x.end.into())
}

#[test]
fn size_serialization() {
    assert_tokens(&size(00), &[Token::U32(00)]);
    assert_tokens(&size(10), &[Token::U32(10)]);
    assert_tokens(&size(20), &[Token::U32(20)]);
    assert_tokens(&size(30), &[Token::U32(30)]);
}

#[test]
fn range_serialization() {
    assert_tokens(
        &range(00..10),
        &[Token::Tuple { len: 2 }, Token::U32(00), Token::U32(10), Token::TupleEnd],
    );
    assert_tokens(
        &range(10..20),
        &[Token::Tuple { len: 2 }, Token::U32(10), Token::U32(20), Token::TupleEnd],
    );
    assert_tokens(
        &range(20..30),
        &[Token::Tuple { len: 2 }, Token::U32(20), Token::U32(30), Token::TupleEnd],
    );
    assert_tokens(
        &range(30..40),
        &[Token::Tuple { len: 2 }, Token::U32(30), Token::U32(40), Token::TupleEnd],
    );
}

#[test]
fn invalid_range_deserialization() {
    assert_tokens::<TextRange>(
        &range(62..92),
        &[Token::Tuple { len: 2 }, Token::U32(62), Token::U32(92), Token::TupleEnd],
    );
    assert_de_tokens_error::<TextRange>(
        &[Token::Tuple { len: 2 }, Token::U32(92), Token::U32(62), Token::TupleEnd],
        "invalid range: 92..62",
    );
}
