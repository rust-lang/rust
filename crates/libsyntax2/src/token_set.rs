use SyntaxKind;

#[derive(Clone, Copy)]
pub(crate) struct TokenSet(pub(crate) u128);

fn mask(kind: SyntaxKind) -> u128 {
    1u128 << (kind as usize)
}

impl TokenSet {
    pub const EMPTY: TokenSet = TokenSet(0);

    pub fn contains(&self, kind: SyntaxKind) -> bool {
        self.0 & mask(kind) != 0
    }
}

#[macro_export]
macro_rules! token_set {
    ($($t:ident),*) => { TokenSet($(1u128 << ($t as usize))|*) };
    ($($t:ident),* ,) => { token_set!($($t),*) };
}

#[macro_export]
macro_rules! token_set_union {
    ($($ts:expr),*) => { TokenSet($($ts.0)|*) };
    ($($ts:expr),* ,) => { token_set_union!($($ts),*) };
}

#[test]
fn token_set_works_for_tokens() {
    use SyntaxKind::*;
    let ts = token_set! { EOF, SHEBANG };
    assert!(ts.contains(EOF));
    assert!(ts.contains(SHEBANG));
    assert!(!ts.contains(PLUS));
}
