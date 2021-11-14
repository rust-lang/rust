use crate::SyntaxKind;

type bits = u64;

pub type IdentKind = u8;

/// Main input to the parser.
///
/// A sequence of tokens represented internally as a struct of arrays.
#[derive(Default)]
pub struct Tokens {
    kind: Vec<SyntaxKind>,
    joint: Vec<bits>,
    ident_kind: Vec<IdentKind>,
}

impl Tokens {
    pub fn push(&mut self, was_joint: bool, kind: SyntaxKind) {
        self.push_impl(was_joint, kind, 0)
    }
    pub fn push_ident(&mut self, ident_kind: IdentKind) {
        self.push_impl(false, SyntaxKind::IDENT, ident_kind)
    }
    fn push_impl(&mut self, was_joint: bool, kind: SyntaxKind, ctx: IdentKind) {
        let idx = self.len();
        if idx % (bits::BITS as usize) == 0 {
            self.joint.push(0);
        }
        if was_joint && idx > 0 {
            self.set_joint(idx - 1);
        }
        self.kind.push(kind);
        self.ident_kind.push(ctx);
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
    pub(crate) fn get(&self, idx: usize) -> (SyntaxKind, bool, IdentKind) {
        if idx > self.len() {
            return self.eof();
        }
        let kind = self.kind[idx];
        let joint = self.get_joint(idx);
        let ident_kind = self.ident_kind[idx];
        (kind, joint, ident_kind)
    }

    #[cold]
    fn eof(&self) -> (SyntaxKind, bool, IdentKind) {
        (SyntaxKind::EOF, false, 0)
    }
}
