use SyntaxKind;

pub(crate) struct TokenSet {
    pub tokens: &'static [SyntaxKind],
}

impl TokenSet {
    pub fn contains(&self, kind: SyntaxKind) -> bool {
        self.tokens.contains(&kind)
    }
}

#[macro_export]
macro_rules! token_set {
    ($($t:ident),*) => {
        TokenSet {
            tokens: &[$($t),*],
        }
    };

    ($($t:ident),* ,) => {
        token_set!($($t),*)
    };
}
