//! Input for the parser -- a sequence of tokens.
//!
//! As of now, parser doesn't have access to the *text* of the tokens, and makes
//! decisions based solely on their classification.

use crate::SyntaxKind;

#[allow(non_camel_case_types)]
type bits = u64;

/// Main input to the parser.
///
/// A sequence of tokens represented internally as a struct of arrays.
#[derive(Default)]
pub struct Tokens {
    kind: Vec<SyntaxKind>,
    joint: Vec<bits>,
    contextual_kind: Vec<SyntaxKind>,
}

impl Tokens {
    #[inline]
    pub fn push(&mut self, kind: SyntaxKind) {
        self.push_impl(kind, SyntaxKind::EOF)
    }
    /// Sets jointness for the last token we've pushed.
    ///
    /// This is a separate API rather than an argument to the `push` to make it
    /// convenient both for textual and mbe tokens. With text, you know whether
    /// the *previous* token was joint, with mbe, you know whether the *current*
    /// one is joint. This API allows for styles of usage:
    ///
    /// ```
    /// // In text:
    /// tokens.was_joint(prev_joint);
    /// tokens.push(curr);
    ///
    /// // In MBE:
    /// token.push(curr);
    /// tokens.push(curr_joint)
    /// ```
    #[inline]
    pub fn was_joint(&mut self) {
        self.set_joint(self.len() - 1);
    }
    #[inline]
    pub fn push_ident(&mut self, contextual_kind: SyntaxKind) {
        self.push_impl(SyntaxKind::IDENT, contextual_kind)
    }
    #[inline]
    fn push_impl(&mut self, kind: SyntaxKind, contextual_kind: SyntaxKind) {
        let idx = self.len();
        if idx % (bits::BITS as usize) == 0 {
            self.joint.push(0);
        }
        self.kind.push(kind);
        self.contextual_kind.push(contextual_kind);
    }
    fn set_joint(&mut self, n: usize) {
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] |= 1 << b_idx;
    }
    fn bit_index(&self, n: usize) -> (usize, usize) {
        let idx = n / (bits::BITS as usize);
        let b_idx = n % (bits::BITS as usize);
        (idx, b_idx)
    }

    fn len(&self) -> usize {
        self.kind.len()
    }
}

/// pub(crate) impl used by the parser.
impl Tokens {
    pub(crate) fn kind(&self, idx: usize) -> SyntaxKind {
        self.kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
    }
    pub(crate) fn contextual_kind(&self, idx: usize) -> SyntaxKind {
        self.contextual_kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
    }
    pub(crate) fn is_joint(&self, n: usize) -> bool {
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] & 1 << b_idx != 0
    }
}
