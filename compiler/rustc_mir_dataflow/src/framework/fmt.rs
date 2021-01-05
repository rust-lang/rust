//! Custom formatting traits used when outputting Graphviz diagrams with the results of a dataflow
//! analysis.

use rustc_index::bit_set::{BitSet, HybridBitSet};
use rustc_index::vec::Idx;
use std::fmt;

/// An extension to `fmt::Debug` for data that can be better printed with some auxiliary data `C`.
pub trait DebugWithContext<C>: Eq + fmt::Debug {
    fn fmt_with(&self, _ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }

    /// Print the difference between `self` and `old`.
    ///
    /// This should print nothing if `self == old`.
    ///
    /// `+` and `-` are typically used to indicate differences. However, these characters are
    /// fairly common and may be needed to print a types representation. If using them to indicate
    /// a diff, prefix them with the "Unit Separator"  control character (␟  U+001F).
    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == old {
            return Ok(());
        }

        write!(f, "\u{001f}+")?;
        self.fmt_with(ctxt, f)?;

        if f.alternate() {
            write!(f, "\n")?;
        } else {
            write!(f, "\t")?;
        }

        write!(f, "\u{001f}-")?;
        old.fmt_with(ctxt, f)
    }
}

/// Implements `fmt::Debug` by deferring to `<T as DebugWithContext<C>>::fmt_with`.
pub struct DebugWithAdapter<'a, T, C> {
    pub this: T,
    pub ctxt: &'a C,
}

impl<T, C> fmt::Debug for DebugWithAdapter<'_, T, C>
where
    T: DebugWithContext<C>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.this.fmt_with(self.ctxt, f)
    }
}

/// Implements `fmt::Debug` by deferring to `<T as DebugWithContext<C>>::fmt_diff_with`.
pub struct DebugDiffWithAdapter<'a, T, C> {
    pub new: T,
    pub old: T,
    pub ctxt: &'a C,
}

impl<T, C> fmt::Debug for DebugDiffWithAdapter<'_, T, C>
where
    T: DebugWithContext<C>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.new.fmt_diff_with(&self.old, self.ctxt, f)
    }
}

// Impls

impl<T, C> DebugWithContext<C> for BitSet<T>
where
    T: Idx + DebugWithContext<C>,
{
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter().map(|i| DebugWithAdapter { this: i, ctxt })).finish()
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = self.domain_size();
        assert_eq!(size, old.domain_size());

        let mut set_in_self = HybridBitSet::new_empty(size);
        let mut cleared_in_self = HybridBitSet::new_empty(size);

        for i in (0..size).map(T::new) {
            match (self.contains(i), old.contains(i)) {
                (true, false) => set_in_self.insert(i),
                (false, true) => cleared_in_self.insert(i),
                _ => continue,
            };
        }

        let mut first = true;
        for idx in set_in_self.iter() {
            let delim = if first {
                "\u{001f}+"
            } else if f.alternate() {
                "\n\u{001f}+"
            } else {
                ", "
            };

            write!(f, "{}", delim)?;
            idx.fmt_with(ctxt, f)?;
            first = false;
        }

        if !f.alternate() {
            first = true;
            if !set_in_self.is_empty() && !cleared_in_self.is_empty() {
                write!(f, "\t")?;
            }
        }

        for idx in cleared_in_self.iter() {
            let delim = if first {
                "\u{001f}-"
            } else if f.alternate() {
                "\n\u{001f}-"
            } else {
                ", "
            };

            write!(f, "{}", delim)?;
            idx.fmt_with(ctxt, f)?;
            first = false;
        }

        Ok(())
    }
}

impl<T, C> DebugWithContext<C> for &'_ T
where
    T: DebugWithContext<C>,
{
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self).fmt_with(ctxt, f)
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self).fmt_diff_with(*old, ctxt, f)
    }
}

impl<C> DebugWithContext<C> for rustc_middle::mir::Local {}
impl<C> DebugWithContext<C> for crate::move_paths::InitIndex {}

impl<'tcx, C> DebugWithContext<C> for crate::move_paths::MovePathIndex
where
    C: crate::move_paths::HasMoveData<'tcx>,
{
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", ctxt.move_data().move_paths[*self])
    }
}

impl<T, C> DebugWithContext<C> for crate::lattice::Dual<T>
where
    T: DebugWithContext<C>,
{
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0).fmt_with(ctxt, f)
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0).fmt_diff_with(&old.0, ctxt, f)
    }
}
