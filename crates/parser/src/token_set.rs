//! A bit-set of `SyntaxKind`s.

use crate::SyntaxKind;

/// A bit-set of `SyntaxKind`s
#[derive(Clone, Copy)]
pub(crate) struct TokenSet([u64; 3]);

const LAST_TOKEN_KIND_DISCRIMINANT: usize = SyntaxKind::SHEBANG as usize;

impl TokenSet {
    pub(crate) const EMPTY: TokenSet = TokenSet([0; 3]);

    pub(crate) const fn new(kinds: &[SyntaxKind]) -> TokenSet {
        let mut res = [0; 3];
        let mut i = 0;
        while i < kinds.len() {
            let kind = kinds[i];
            debug_assert!(
                kind as usize <= LAST_TOKEN_KIND_DISCRIMINANT,
                "Expected a token `SyntaxKind`"
            );
            let idx = kind as usize / 64;
            res[idx] |= mask(kind);
            i += 1;
        }
        TokenSet(res)
    }

    pub(crate) const fn union(self, other: TokenSet) -> TokenSet {
        TokenSet([self.0[0] | other.0[0], self.0[1] | other.0[1], self.0[2] | other.0[2]])
    }

    pub(crate) const fn contains(&self, kind: SyntaxKind) -> bool {
        debug_assert!(
            kind as usize <= LAST_TOKEN_KIND_DISCRIMINANT,
            "Expected a token `SyntaxKind`"
        );
        let idx = kind as usize / 64;
        self.0[idx] & mask(kind) != 0
    }
}

const fn mask(kind: SyntaxKind) -> u64 {
    debug_assert!(kind as usize <= LAST_TOKEN_KIND_DISCRIMINANT, "Expected a token `SyntaxKind`");
    1 << (kind as usize % 64)
}

#[test]
fn token_set_works_for_tokens() {
    use crate::SyntaxKind::*;
    let ts = TokenSet::new(&[EOF, SHEBANG]);
    assert!(ts.contains(EOF));
    assert!(ts.contains(SHEBANG));
    assert!(!ts.contains(PLUS));
}
