use rustc_hir as hir;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
pub enum BoundConstness {
    /// `T: Trait`
    NotConst,
    /// `T: ~const Trait`
    ///
    /// Requires resolving to const only when we are in a const context.
    ConstIfConst,
}

impl BoundConstness {
    /// Reduce `self` and `constness` to two possible combined states instead of four.
    pub fn and(&mut self, constness: hir::Constness) -> hir::Constness {
        match (constness, self) {
            (hir::Constness::Const, BoundConstness::ConstIfConst) => hir::Constness::Const,
            (_, this) => {
                *this = BoundConstness::NotConst;
                hir::Constness::NotConst
            }
        }
    }
}

impl fmt::Display for BoundConstness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConst => f.write_str("normal"),
            Self::ConstIfConst => f.write_str("`~const`"),
        }
    }
}
