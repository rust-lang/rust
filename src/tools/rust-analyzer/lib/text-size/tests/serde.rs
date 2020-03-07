use {serde_test::*, text_size::*};

#[test]
fn size() {
    assert_tokens(&TextSize::new(00), &[Token::U32(00)]);
    assert_tokens(&TextSize::new(10), &[Token::U32(10)]);
    assert_tokens(&TextSize::new(20), &[Token::U32(20)]);
    assert_tokens(&TextSize::new(30), &[Token::U32(30)]);
}

#[test]
fn range() {
    assert_tokens(
        &TextRange::from(00..10),
        &[
            Token::Tuple { len: 2 },
            Token::U32(00),
            Token::U32(10),
            Token::TupleEnd,
        ],
    );
    assert_tokens(
        &TextRange::from(10..20),
        &[
            Token::Tuple { len: 2 },
            Token::U32(10),
            Token::U32(20),
            Token::TupleEnd,
        ],
    );
    assert_tokens(
        &TextRange::from(20..30),
        &[
            Token::Tuple { len: 2 },
            Token::U32(20),
            Token::U32(30),
            Token::TupleEnd,
        ],
    );
    assert_tokens(
        &TextRange::from(30..40),
        &[
            Token::Tuple { len: 2 },
            Token::U32(30),
            Token::U32(40),
            Token::TupleEnd,
        ],
    );
}
