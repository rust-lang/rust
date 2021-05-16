extern crate test;
use crate::parser::{TokenCursor, TokenCursorFrame};
use crate::tokenstream::DelimSpan;
use rustc_ast::token::DelimToken;
use rustc_span::DUMMY_SP;
use std::hint::black_box;
use test::Bencher;

fn mk_dummy_token_cursor_frame() -> TokenCursorFrame {
    TokenCursorFrame::new(DelimSpan::from_single(DUMMY_SP), DelimToken::Paren, Default::default())
}

fn mk_token_cursor(n: usize) -> TokenCursor<true> {
    TokenCursor {
        frame: mk_dummy_token_cursor_frame(),
        stack: vec![mk_dummy_token_cursor_frame(); n],
        nncablt: Default::default(),
    }
}
macro_rules! bench_for_n {
    ($name: ident, $n: expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let tc = black_box(mk_token_cursor($n));
            b.iter(|| {
                let _ = tc.clone();
            });
        }
    };
}

bench_for_n!(token_cursor_clone_1, 1);
bench_for_n!(token_cursor_clone_2, 2);
bench_for_n!(token_cursor_clone_4, 4);
bench_for_n!(token_cursor_clone_8, 8);
bench_for_n!(token_cursor_clone_16, 16);
bench_for_n!(token_cursor_clone_32, 32);
bench_for_n!(token_cursor_clone_64, 64);
bench_for_n!(token_cursor_clone_128, 128);
bench_for_n!(token_cursor_clone_256, 256);
