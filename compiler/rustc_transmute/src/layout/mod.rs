use std::fmt::{self, Debug};
use std::hash::Hash;
use std::ops::RangeInclusive;

pub(crate) mod tree;
pub(crate) use tree::Tree;

pub(crate) mod dfa;
pub(crate) use dfa::Dfa;

#[derive(Debug)]
pub(crate) struct Uninhabited;

/// A range of byte values, or the uninit byte.
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub(crate) struct Byte {
    // An inclusive-inclusive range. We use this instead of `RangeInclusive`
    // because `RangeInclusive: !Copy`.
    //
    // `None` means uninit.
    //
    // FIXME(@joshlf): Optimize this representation. Some pairs of values (where
    // `lo > hi`) are illegal, and we could use these to represent `None`.
    range: Option<(u8, u8)>,
}

impl Byte {
    fn new(range: RangeInclusive<u8>) -> Self {
        Self { range: Some((*range.start(), *range.end())) }
    }

    fn from_val(val: u8) -> Self {
        Self { range: Some((val, val)) }
    }

    pub(crate) fn uninit() -> Byte {
        Byte { range: None }
    }

    /// Returns `None` if `self` is the uninit byte.
    pub(crate) fn range(&self) -> Option<RangeInclusive<u8>> {
        self.range.map(|(lo, hi)| lo..=hi)
    }

    /// Are any of the values in `self` transmutable into `other`?
    ///
    /// Note two special cases: An uninit byte is only transmutable into another
    /// uninit byte. Any byte is transmutable into an uninit byte.
    pub(crate) fn transmutable_into(&self, other: &Byte) -> bool {
        match (self.range, other.range) {
            (None, None) => true,
            (None, Some(_)) => false,
            (Some(_), None) => true,
            (Some((slo, shi)), Some((olo, ohi))) => slo <= ohi && olo <= shi,
        }
    }
}

impl fmt::Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.range {
            None => write!(f, "uninit"),
            Some((lo, hi)) => write!(f, "{lo}..={hi}"),
        }
    }
}

#[cfg(test)]
impl From<u8> for Byte {
    fn from(src: u8) -> Self {
        Self::from_val(src)
    }
}

pub(crate) trait Def: Debug + Hash + Eq + PartialEq + Copy + Clone {
    fn has_safety_invariants(&self) -> bool;
}
pub trait Ref: Debug + Hash + Eq + PartialEq + Copy + Clone {
    fn min_align(&self) -> usize;

    fn size(&self) -> usize;

    fn is_mutable(&self) -> bool;
}

impl Def for ! {
    fn has_safety_invariants(&self) -> bool {
        unreachable!()
    }
}

impl Ref for ! {
    fn min_align(&self) -> usize {
        unreachable!()
    }
    fn size(&self) -> usize {
        unreachable!()
    }
    fn is_mutable(&self) -> bool {
        unreachable!()
    }
}

#[cfg(test)]
impl<const N: usize> Ref for [(); N] {
    fn min_align(&self) -> usize {
        N
    }

    fn size(&self) -> usize {
        N
    }

    fn is_mutable(&self) -> bool {
        false
    }
}

#[cfg(feature = "rustc")]
pub mod rustc {
    use std::fmt::{self, Write};

    use rustc_abi::Layout;
    use rustc_middle::mir::Mutability;
    use rustc_middle::ty::layout::{HasTyCtxt, LayoutCx, LayoutError};
    use rustc_middle::ty::{self, Ty};

    /// A reference in the layout.
    #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
    pub struct Ref<'tcx> {
        pub lifetime: ty::Region<'tcx>,
        pub ty: Ty<'tcx>,
        pub mutability: Mutability,
        pub align: usize,
        pub size: usize,
    }

    impl<'tcx> super::Ref for Ref<'tcx> {
        fn min_align(&self) -> usize {
            self.align
        }

        fn size(&self) -> usize {
            self.size
        }

        fn is_mutable(&self) -> bool {
            match self.mutability {
                Mutability::Mut => true,
                Mutability::Not => false,
            }
        }
    }
    impl<'tcx> Ref<'tcx> {}

    impl<'tcx> fmt::Display for Ref<'tcx> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_char('&')?;
            if self.mutability == Mutability::Mut {
                f.write_str("mut ")?;
            }
            self.ty.fmt(f)
        }
    }

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

    pub(crate) fn layout_of<'tcx>(
        cx: LayoutCx<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
        use rustc_middle::ty::layout::LayoutOf;
        let ty = cx.tcx().erase_regions(ty);
        cx.layout_of(ty).map(|tl| tl.layout)
    }
}
