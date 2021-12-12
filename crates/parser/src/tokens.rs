use crate::SyntaxKind;

#[allow(non_camel_case_types)]
type bits = u64;

/// `Token` abstracts the cursor of `TokenSource` operates on.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct Token {
    /// What is the current token?
    pub(crate) kind: SyntaxKind,

    /// Is the current token joined to the next one (`> >` vs `>>`).
    pub(crate) is_jointed_to_next: bool,
    pub(crate) contextual_kw: SyntaxKind,
}

/// Main input to the parser.
///
/// A sequence of tokens represented internally as a struct of arrays.
#[derive(Default)]
pub struct Tokens {
    kind: Vec<SyntaxKind>,
    joint: Vec<bits>,
    contextual_kw: Vec<SyntaxKind>,
}

impl Tokens {
    pub fn push(&mut self, was_joint: bool, kind: SyntaxKind) {
        self.push_impl(was_joint, kind, SyntaxKind::EOF)
    }
    pub fn push_ident(&mut self, contextual_kw: SyntaxKind) {
        self.push_impl(false, SyntaxKind::IDENT, contextual_kw)
    }
    fn push_impl(&mut self, was_joint: bool, kind: SyntaxKind, contextual_kw: SyntaxKind) {
        let idx = self.len();
        if idx % (bits::BITS as usize) == 0 {
            self.joint.push(0);
        }
        if was_joint && idx > 0 {
            self.set_joint(idx - 1);
        }
        self.kind.push(kind);
        self.contextual_kw.push(contextual_kw);
    }
    fn set_joint(&mut self, n: usize) {
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] |= 1 << b_idx;
    }
    fn get_joint(&self, n: usize) -> bool {
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] & 1 << b_idx != 0
    }
    fn bit_index(&self, n: usize) -> (usize, usize) {
        let idx = n / (bits::BITS as usize);
        let b_idx = n % (bits::BITS as usize);
        (idx, b_idx)
    }

    pub fn len(&self) -> usize {
        self.kind.len()
    }
    pub(crate) fn get(&self, idx: usize) -> Token {
        if idx < self.len() {
            let kind = self.kind[idx];
            let is_jointed_to_next = self.get_joint(idx);
            let contextual_kw = self.contextual_kw[idx];
            Token { kind, is_jointed_to_next, contextual_kw }
        } else {
            self.eof()
        }
    }

    #[cold]
    fn eof(&self) -> Token {
        Token { kind: SyntaxKind::EOF, is_jointed_to_next: false, contextual_kw: SyntaxKind::EOF }
    }
}
