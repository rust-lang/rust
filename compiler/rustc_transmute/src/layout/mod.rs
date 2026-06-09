use std::fmt::{self, Debug};
use std::hash::Hash;
use std::ops::RangeInclusive;

pub(crate) mod tree;
pub(crate) use tree::Tree;

pub(crate) mod dfa;
pub(crate) use dfa::{Dfa, union};

#[derive(Debug)]
pub(crate) struct Uninhabited;

/// A range of byte values (including an uninit byte value).
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub(crate) struct Byte {
    // An inclusive-exclusive range. We use this instead of `Range` because `Range: !Copy`.
    //
    // Uninit byte value is represented by 256.
    pub(crate) start: u16,
    pub(crate) end: u16,
}

impl Byte {
    const UNINIT: u16 = 256;

    #[inline]
    fn new(range: RangeInclusive<u8>) -> Self {
        let start: u16 = (*range.start()).into();
        let end: u16 = (*range.end()).into();
        Byte { start, end: end + 1 }
    }

    #[inline]
    fn from_val(val: u8) -> Self {
        let val: u16 = val.into();
        Byte { start: val, end: val + 1 }
    }

    #[inline]
    fn uninit() -> Byte {
        Byte { start: 0, end: Self::UNINIT + 1 }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    #[inline]
    fn contains_uninit(&self) -> bool {
        self.start <= Self::UNINIT && Self::UNINIT < self.end
    }
}

impl fmt::Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == Self::UNINIT && self.end == Self::UNINIT + 1 {
            write!(f, "uninit")
        } else if self.start <= Self::UNINIT && self.end == Self::UNINIT + 1 {
            write!(f, "{}..{}|uninit", self.start, self.end - 1)
        } else {
            write!(f, "{}..{}", self.start, self.end)
        }
    }
}

impl From<RangeInclusive<u8>> for Byte {
    fn from(src: RangeInclusive<u8>) -> Self {
        Self::new(src)
    }
}

impl From<u8> for Byte {
    #[inline]
    fn from(src: u8) -> Self {
        Self::from_val(src)
    }
}

/// A reference, i.e., `&'region T` or `&'region mut T`.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub(crate) struct Reference<R, T>
where
    R: Region,
    T: Type,
{
    pub(crate) region: R,
    pub(crate) is_mut: bool,
    pub(crate) referent: T,
    pub(crate) referent_size: usize,
    pub(crate) referent_align: usize,
}

impl<R, T> fmt::Display for Reference<R, T>
where
    R: Region,
    T: Type,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("&")?;
        if self.is_mut {
            f.write_str("mut ")?;
        }
        self.referent.fmt(f)
    }
}

pub(crate) trait Def: Debug + Hash + Eq + PartialEq + Copy + Clone {
    fn has_safety_invariants(&self) -> bool;
}

pub(crate) trait Region: Debug + Hash + Eq + PartialEq + Copy + Clone {}

pub(crate) trait Type: Debug + Hash + Eq + PartialEq + Copy + Clone {}

impl Def for ! {
    fn has_safety_invariants(&self) -> bool {
        unreachable!()
    }
}

impl Region for ! {}

impl Type for ! {}

#[cfg(test)]
impl Region for usize {}

#[cfg(test)]
impl Type for () {}

#[cfg(feature = "rustc")]
pub mod rustc {
    use rustc_abi::Layout;
    use rustc_middle::ty::layout::{HasTyCtxt, LayoutCx, LayoutError};
    use rustc_middle::ty::{self, Region, Ty};

    /// A visibility node in the layout.
    #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
    pub enum Def<'tcx> {
        Adt(ty::AdtDef<'tcx>),
        Variant(&'tcx ty::VariantDef),
        Field(&'tcx ty::FieldDef),
        Primitive,
    }

    impl<'tcx> super::Def for Def<'tcx> {
        fn has_safety_invariants(&self) -> bool {
            // Rust presently has no notion of 'unsafe fields', so for now we
            // make the conservative assumption that everything besides
            // primitive types carry safety invariants.
            self != &Self::Primitive
        }
    }

    impl<'tcx> super::Region for Region<'tcx> {}

    impl<'tcx> super::Type for Ty<'tcx> {}

    pub(crate) fn layout_of<'tcx>(
        cx: LayoutCx<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
        use rustc_middle::ty::layout::LayoutOf;
        let ty = cx.tcx().erase_and_anonymize_regions(ty);
        cx.layout_of(ty).map(|tl| tl.layout)
    }
}
