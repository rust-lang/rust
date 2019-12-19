//! A bit-set of `SyntaxKind`s.

use crate::SyntaxKind;

/// A bit-set of `SyntaxKind`s
#[derive(Clone, Copy)]
pub(crate) struct TokenSet(u128);

impl TokenSet {
    pub(crate) const EMPTY: TokenSet = TokenSet(0);

    pub(crate) const fn singleton(kind: SyntaxKind) -> TokenSet {
        TokenSet(mask(kind))
    }

    pub(crate) const fn union(self, other: TokenSet) -> TokenSet {
        TokenSet(self.0 | other.0)
    }

    pub(crate) fn contains(&self, kind: SyntaxKind) -> bool {
        self.0 & mask(kind) != 0
    }
}

const fn mask(kind: SyntaxKind) -> u128 {
    1u128 << (kind as usize)
}

#[macro_export]
macro_rules! token_set {
    ($($t:expr),*) => { TokenSet::EMPTY$(.union(TokenSet::singleton($t)))* };
    ($($t:expr),* ,) => { token_set!($($t),*) };
}

#[test]
fn token_set_works_for_tokens() {
    use crate::SyntaxKind::*;
    let ts = token_set![EOF, SHEBANG];
    assert!(ts.contains(EOF));
    assert!(ts.contains(SHEBANG));
    assert!(!ts.contains(PLUS));
}
