use std::fmt::{Debug, Formatter};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub trait Lattice: Clone {
    fn bottom() -> Self;
    fn join(&mut self, other: &Self) -> bool;
}

/// Extend the type with a Top point.
#[derive(Clone, PartialEq)]
pub enum WTop<T> {
    Top,
    Value(T)
}

impl<T: Lattice> Lattice for WTop<T> {
    fn bottom() -> Self {
        WTop::Value(<T as Lattice>::bottom())
    }

    /// V + V = join(v, v)
    /// ⊤ + V = ⊤ (no change)
    /// V + ⊤ = ⊤
    /// ⊤ + ⊤ = ⊤ (no change)
    fn join(&mut self, other: &Self) -> bool {
        match (self, other) {
            (&mut WTop::Value(ref mut this), &WTop::Value(ref o)) => <T as Lattice>::join(this, o),
            (&mut WTop::Top, _) => false,
            (this, &WTop::Top) => {
                *this = WTop::Top;
                true
            }
        }
    }
}

impl<T: Debug> Debug for WTop<T> {
    fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
        match *self {
            WTop::Top => f.write_str("⊤"),
            WTop::Value(ref t) => <T as Debug>::fmt(t, f)
        }
    }
}

/// Extend the type with a bottom point
///
/// This guarantees the bottom() of the underlying lattice won’t get called so it may be
/// implemented as a `panic!()` or something.
#[derive(Clone, PartialEq)]
pub enum WBottom<T> {
    Bottom,
    Value(T)
}

impl<T: Lattice> Lattice for WBottom<T> {
    fn bottom() -> Self {
        WBottom::Bottom
    }

    /// V + V = join(v, v)
    /// ⊥ + V = V
    /// V + ⊥ = V (no change)
    /// ⊥ + ⊥ = ⊥ (no change)
    fn join(&mut self, other: &Self) -> bool {
        match (self, other) {
            (&mut WBottom::Value(ref mut this), &WBottom::Value(ref o)) =>
                <T as Lattice>::join(this, o),
            (_, &WBottom::Bottom) => false,
            (this, o) => {
                *this = o.clone();
                true
            }
        }
    }

}

impl<T: Debug> Debug for WBottom<T> {
    fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
        match *self {
            WBottom::Bottom => f.write_str("⊥"),
            WBottom::Value(ref t) => <T as Debug>::fmt(t, f)
        }
    }
}

/// Extend the type with both bottom and top points.
type WTopBottom<T> = WTop<WBottom<T>>;

impl<K, T, H> Lattice for HashMap<K, T, H>
where K: Clone + Eq + ::std::hash::Hash,
      T: Lattice,
      H: Clone + ::std::hash::BuildHasher + ::std::default::Default
{
    fn bottom() -> Self {
        HashMap::default()
    }
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (key, val) in other.iter() {
            match self.entry(key.clone()) {
                Entry::Vacant(e) => {
                    e.insert(val.clone());
                    changed = true
                }
                Entry::Occupied(mut e) => changed |= e.get_mut().join(val)
            }
        }
        changed
    }
}


