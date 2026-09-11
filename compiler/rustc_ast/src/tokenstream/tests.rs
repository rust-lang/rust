use rustc_span::DUMMY_SP;

use crate::token::{Delimiter, Token, TokenKind};
use crate::tokenarena::{ArenaTokenStreamBuilder, DelimitedData};
use crate::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenCursor, TokenStream};

#[test]
fn test_token_stream_iter() {
    let ts = TokenStream::token_alone(TokenKind::Eq, DUMMY_SP);
    assert_eq!(ts.len(), 1);

    let iter = ts.iter();
    assert_eq!(iter.size_hint(), (1, Some(1)));
}

#[test]
fn foo() {
    let mut arena = ArenaTokenStreamBuilder::default();
    let open1 = arena.start_delimited();
    arena.push_token_alone(Token::new(TokenKind::Plus, DUMMY_SP));
    let open2 = arena.start_delimited();
    arena.push_token_alone(Token::new(TokenKind::Plus, DUMMY_SP));
    arena.finish_delimited(
        open2,
        DelimitedData {
            span: DelimSpan::from_single(DUMMY_SP),
            spacing: DelimSpacing { open: Spacing::Alone, close: Spacing::Alone },
            delimiter: Delimiter::Parenthesis,
        },
    );
    arena.finish_delimited(
        open1,
        DelimitedData {
            span: DelimSpan::from_single(DUMMY_SP),
            spacing: DelimSpacing { open: Spacing::Alone, close: Spacing::Alone },
            delimiter: Delimiter::Parenthesis,
        },
    );

    let mut cursor = TokenCursor::new(arena);
    for _ in 0..100 {
        cursor.next_and_bump();
    }
}
