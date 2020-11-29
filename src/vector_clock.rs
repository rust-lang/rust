use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::Idx;
use smallvec::SmallVec;
use std::{
    cmp::Ordering,
    convert::TryFrom,
    fmt::{self, Debug},
    mem,
    ops::Index,
};

/// A vector clock index, this is associated with a thread id
/// but in some cases one vector index may be shared with
/// multiple thread ids if it safe to do so.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct VectorIdx(u32);

impl VectorIdx {
    #[inline(always)]
    pub fn to_u32(self) -> u32 {
        self.0
    }

    pub const MAX_INDEX: VectorIdx = VectorIdx(u32::MAX);
}

impl Idx for VectorIdx {
    #[inline]
    fn new(idx: usize) -> Self {
        VectorIdx(u32::try_from(idx).unwrap())
    }

    #[inline]
    fn index(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl From<u32> for VectorIdx {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// A sparse mapping of vector index values to vector clocks, this
/// is optimized for the common case with only one element stored
/// inside the map.
/// This is used to store the set of currently active release
/// sequences at a given memory location, since RMW operations
/// allow for multiple release sequences to be active at once
/// and to be collapsed back to one active release sequence
/// once a non RMW atomic store operation occurs.
/// An all zero vector is considered to be equal to no
/// element stored internally since it will never be
/// stored and has no meaning as a release sequence
/// vector clock.
#[derive(Clone)]
pub struct VSmallClockMap(VSmallClockMapInner);

#[derive(Clone)]
enum VSmallClockMapInner {
    /// Zero or 1 vector elements, common
    /// case for the sparse set.
    /// The all zero vector clock is treated
    /// as equal to the empty element.
    Small(VectorIdx, VClock),

    /// Hash-map of vector clocks.
    Large(FxHashMap<VectorIdx, VClock>),
}

impl VSmallClockMap {
    /// Remove all clock vectors from the map, setting them
    /// to the zero vector.
    pub fn clear(&mut self) {
        match &mut self.0 {
            VSmallClockMapInner::Small(_, clock) => clock.set_zero_vector(),
            VSmallClockMapInner::Large(hash_map) => {
                hash_map.clear();
            }
        }
    }

    /// Remove all clock vectors except for the clock vector
    /// stored at the given index, which is retained.
    pub fn retain_index(&mut self, index: VectorIdx) {
        match &mut self.0 {
            VSmallClockMapInner::Small(small_idx, clock) => {
                if index != *small_idx {
                    // The zero-vector is considered to equal
                    // the empty element.
                    clock.set_zero_vector()
                }
            }
            VSmallClockMapInner::Large(hash_map) => {
                let value = hash_map.remove(&index).unwrap_or_default();
                self.0 = VSmallClockMapInner::Small(index, value);
            }
        }
    }

    /// Insert the vector clock into the associated vector
    /// index.
    pub fn insert(&mut self, index: VectorIdx, clock: &VClock) {
        match &mut self.0 {
            VSmallClockMapInner::Small(small_idx, small_clock) => {
                if small_clock.is_zero_vector() {
                    *small_idx = index;
                    small_clock.clone_from(clock);
                } else if !clock.is_zero_vector() {
                    // Convert to using the hash-map representation.
                    let mut hash_map = FxHashMap::default();
                    hash_map.insert(*small_idx, mem::take(small_clock));
                    hash_map.insert(index, clock.clone());
                    self.0 = VSmallClockMapInner::Large(hash_map);
                }
            }
            VSmallClockMapInner::Large(hash_map) =>
                if !clock.is_zero_vector() {
                    hash_map.insert(index, clock.clone());
                },
        }
    }

    /// Try to load the vector clock associated with the current
    ///  vector index.
    pub fn get(&self, index: VectorIdx) -> Option<&VClock> {
        match &self.0 {
            VSmallClockMapInner::Small(small_idx, small_clock) => {
                if *small_idx == index && !small_clock.is_zero_vector() {
                    Some(small_clock)
                } else {
                    None
                }
            }
            VSmallClockMapInner::Large(hash_map) => hash_map.get(&index),
        }
    }
}

impl Default for VSmallClockMap {
    #[inline]
    fn default() -> Self {
        VSmallClockMap(VSmallClockMapInner::Small(VectorIdx::new(0), VClock::default()))
    }
}

impl Debug for VSmallClockMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print the contents of the small vector clock set as the map
        // of vector index to vector clock that they represent.
        let mut map = f.debug_map();
        match &self.0 {
            VSmallClockMapInner::Small(small_idx, small_clock) =>
                if !small_clock.is_zero_vector() {
                    map.entry(&small_idx, &small_clock);
                },
            VSmallClockMapInner::Large(hash_map) =>
                for (idx, elem) in hash_map.iter() {
                    map.entry(idx, elem);
                },
        }
        map.finish()
    }
}

impl PartialEq for VSmallClockMap {
    fn eq(&self, other: &Self) -> bool {
        use VSmallClockMapInner::*;
        match (&self.0, &other.0) {
            (Small(i1, c1), Small(i2, c2)) => {
                if c1.is_zero_vector() {
                    // Either they are both zero or they are non-equal
                    c2.is_zero_vector()
                } else {
                    // At least one is non-zero, so the full comparison is correct
                    i1 == i2 && c1 == c2
                }
            }
            (Small(idx, clock), Large(hash_map)) | (Large(hash_map), Small(idx, clock)) => {
                if hash_map.len() == 0 {
                    // Equal to the empty hash-map
                    clock.is_zero_vector()
                } else if hash_map.len() == 1 {
                    // Equal to the hash-map with one element
                    let (hash_idx, hash_clock) = hash_map.iter().next().unwrap();
                    hash_idx == idx && hash_clock == clock
                } else {
                    false
                }
            }
            (Large(map1), Large(map2)) => map1 == map2,
        }
    }
}

impl Eq for VSmallClockMap {}

/// The size of the vector-clock to store inline
/// clock vectors larger than this will be stored on the heap
const SMALL_VECTOR: usize = 4;

/// The type of the time-stamps recorded in the data-race detector
/// set to a type of unsigned integer
pub type VTimestamp = u32;

/// A vector clock for detecting data-races, this is conceptually
/// a map from a vector index (and thus a thread id) to a timestamp.
/// The compare operations require that the invariant that the last
/// element in the internal timestamp slice must not be a 0, hence
/// all zero vector clocks are always represented by the empty slice;
/// and allows for the implementation of compare operations to short
/// circuit the calculation and return the correct result faster,
/// also this means that there is only one unique valid length
/// for each set of vector clock values and hence the PartialEq
//  and Eq derivations are correct.
#[derive(PartialEq, Eq, Default, Debug)]
pub struct VClock(SmallVec<[VTimestamp; SMALL_VECTOR]>);

impl VClock {
    /// Create a new vector-clock containing all zeros except
    /// for a value at the given index
    pub fn new_with_index(index: VectorIdx, timestamp: VTimestamp) -> VClock {
        let len = index.index() + 1;
        let mut vec = smallvec::smallvec![0; len];
        vec[index.index()] = timestamp;
        VClock(vec)
    }

    /// Load the internal timestamp slice in the vector clock
    #[inline]
    pub fn as_slice(&self) -> &[VTimestamp] {
        self.0.as_slice()
    }

    /// Get a mutable slice to the internal vector with minimum `min_len`
    /// elements, to preserve invariants this vector must modify
    /// the `min_len`-1 nth element to a non-zero value
    #[inline]
    fn get_mut_with_min_len(&mut self, min_len: usize) -> &mut [VTimestamp] {
        if self.0.len() < min_len {
            self.0.resize(min_len, 0);
        }
        assert!(self.0.len() >= min_len);
        self.0.as_mut_slice()
    }

    /// Increment the vector clock at a known index
    /// this will panic if the vector index overflows
    #[inline]
    pub fn increment_index(&mut self, idx: VectorIdx) {
        let idx = idx.index();
        let mut_slice = self.get_mut_with_min_len(idx + 1);
        let idx_ref = &mut mut_slice[idx];
        *idx_ref = idx_ref.checked_add(1).expect("Vector clock overflow")
    }

    // Join the two vector-clocks together, this
    // sets each vector-element to the maximum value
    // of that element in either of the two source elements.
    pub fn join(&mut self, other: &Self) {
        let rhs_slice = other.as_slice();
        let lhs_slice = self.get_mut_with_min_len(rhs_slice.len());
        for (l, &r) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            *l = r.max(*l);
        }
    }

    /// Set the element at the current index of the vector
    pub fn set_at_index(&mut self, other: &Self, idx: VectorIdx) {
        let idx = idx.index();
        let mut_slice = self.get_mut_with_min_len(idx + 1);
        let slice = other.as_slice();
        mut_slice[idx] = slice[idx];
    }

    /// Set the vector to the all-zero vector
    #[inline]
    pub fn set_zero_vector(&mut self) {
        self.0.clear();
    }

    /// Return if this vector is the all-zero vector
    pub fn is_zero_vector(&self) -> bool {
        self.0.is_empty()
    }
}

impl Clone for VClock {
    fn clone(&self) -> Self {
        VClock(self.0.clone())
    }

    // Optimized clone-from, can be removed
    // and replaced with a derive once a similar
    // optimization is inserted into SmallVec's
    // clone implementation.
    fn clone_from(&mut self, source: &Self) {
        let source_slice = source.as_slice();
        self.0.clear();
        self.0.extend_from_slice(source_slice);
    }
}

impl PartialOrd for VClock {
    fn partial_cmp(&self, other: &VClock) -> Option<Ordering> {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // Iterate through the combined vector slice continuously updating
        // the value of `order` to the current comparison of the vector from
        // index 0 to the currently checked index.
        // An Equal ordering can be converted into Less or Greater ordering
        // on finding an element that is less than or greater than the other
        // but if one Greater and one Less element-wise comparison is found
        // then no ordering is possible and so directly return an ordering
        // of None.
        let mut iter = lhs_slice.iter().zip(rhs_slice.iter());
        let mut order = match iter.next() {
            Some((lhs, rhs)) => lhs.cmp(rhs),
            None => Ordering::Equal,
        };
        for (l, r) in iter {
            match order {
                Ordering::Equal => order = l.cmp(r),
                Ordering::Less =>
                    if l > r {
                        return None;
                    },
                Ordering::Greater =>
                    if l < r {
                        return None;
                    },
            }
        }

        // Now test if either left or right have trailing elements,
        // by the invariant the trailing elements have at least 1
        // non zero value, so no additional calculation is required
        // to determine the result of the PartialOrder.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        match l_len.cmp(&r_len) {
            // Equal means no additional elements: return current order
            Ordering::Equal => Some(order),
            // Right has at least 1 element > than the implicit 0,
            // so the only valid values are Ordering::Less or None.
            Ordering::Less => match order {
                Ordering::Less | Ordering::Equal => Some(Ordering::Less),
                Ordering::Greater => None,
            },
            // Left has at least 1 element > than the implicit 0,
            // so the only valid values are Ordering::Greater or None.
            Ordering::Greater => match order {
                Ordering::Greater | Ordering::Equal => Some(Ordering::Greater),
                Ordering::Less => None,
            },
        }
    }

    fn lt(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If l_len > r_len then at least one element
        // in l_len is > than r_len, therefore the result
        // is either Some(Greater) or None, so return false
        // early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len <= r_len {
            // If any elements on the left are greater than the right
            // then the result is None or Some(Greater), both of which
            // return false, the earlier test asserts that no elements in the
            // extended tail violate this assumption. Otherwise l <= r, finally
            // the case where the values are potentially equal needs to be considered
            // and false returned as well
            let mut equal = l_len == r_len;
            for (&l, &r) in lhs_slice.iter().zip(rhs_slice.iter()) {
                if l > r {
                    return false;
                } else if l < r {
                    equal = false;
                }
            }
            !equal
        } else {
            false
        }
    }

    fn le(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If l_len > r_len then at least one element
        // in l_len is > than r_len, therefore the result
        // is either Some(Greater) or None, so return false
        // early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len <= r_len {
            // If any elements on the left are greater than the right
            // then the result is None or Some(Greater), both of which
            // return false, the earlier test asserts that no elements in the
            // extended tail violate this assumption. Otherwise l <= r
            !lhs_slice.iter().zip(rhs_slice.iter()).any(|(&l, &r)| l > r)
        } else {
            false
        }
    }

    fn gt(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If r_len > l_len then at least one element
        // in r_len is > than l_len, therefore the result
        // is either Some(Less) or None, so return false
        // early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len >= r_len {
            // If any elements on the left are less than the right
            // then the result is None or Some(Less), both of which
            // return false, the earlier test asserts that no elements in the
            // extended tail violate this assumption. Otherwise l >=, finally
            // the case where the values are potentially equal needs to be considered
            // and false returned as well
            let mut equal = l_len == r_len;
            for (&l, &r) in lhs_slice.iter().zip(rhs_slice.iter()) {
                if l < r {
                    return false;
                } else if l > r {
                    equal = false;
                }
            }
            !equal
        } else {
            false
        }
    }

    fn ge(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If r_len > l_len then at least one element
        // in r_len is > than l_len, therefore the result
        // is either Some(Less) or None, so return false
        // early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len >= r_len {
            // If any elements on the left are less than the right
            // then the result is None or Some(Less), both of which
            // return false, the earlier test asserts that no elements in the
            // extended tail violate this assumption. Otherwise l >= r
            !lhs_slice.iter().zip(rhs_slice.iter()).any(|(&l, &r)| l < r)
        } else {
            false
        }
    }
}

impl Index<VectorIdx> for VClock {
    type Output = VTimestamp;

    #[inline]
    fn index(&self, index: VectorIdx) -> &VTimestamp {
        self.as_slice().get(index.to_u32() as usize).unwrap_or(&0)
    }
}

/// Test vector clock ordering operations
///  data-race detection is tested in the external
///  test suite
#[cfg(test)]
mod tests {

    use super::{VClock, VSmallClockMap, VTimestamp, VectorIdx};
    use std::cmp::Ordering;

    #[test]
    fn test_equal() {
        let mut c1 = VClock::default();
        let mut c2 = VClock::default();
        assert_eq!(c1, c2);
        c1.increment_index(VectorIdx(5));
        assert_ne!(c1, c2);
        c2.increment_index(VectorIdx(53));
        assert_ne!(c1, c2);
        c1.increment_index(VectorIdx(53));
        assert_ne!(c1, c2);
        c2.increment_index(VectorIdx(5));
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_partial_order() {
        // Small test
        assert_order(&[1], &[1], Some(Ordering::Equal));
        assert_order(&[1], &[2], Some(Ordering::Less));
        assert_order(&[2], &[1], Some(Ordering::Greater));
        assert_order(&[1], &[1, 2], Some(Ordering::Less));
        assert_order(&[2], &[1, 2], None);

        // Misc tests
        assert_order(&[400], &[0, 1], None);

        // Large test
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0],
            Some(Ordering::Equal),
        );
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0],
            Some(Ordering::Less),
        );
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0],
            Some(Ordering::Greater),
        );
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0],
            None,
        );
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0],
            Some(Ordering::Less),
        );
        assert_order(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0],
            Some(Ordering::Less),
        );
    }

    fn from_slice(mut slice: &[VTimestamp]) -> VClock {
        while let Some(0) = slice.last() {
            slice = &slice[..slice.len() - 1]
        }
        VClock(smallvec::SmallVec::from_slice(slice))
    }

    fn assert_order(l: &[VTimestamp], r: &[VTimestamp], o: Option<Ordering>) {
        let l = from_slice(l);
        let r = from_slice(r);

        //Test partial_cmp
        let compare = l.partial_cmp(&r);
        assert_eq!(compare, o, "Invalid comparison\n l: {:?}\n r: {:?}", l, r);
        let alt_compare = r.partial_cmp(&l);
        assert_eq!(
            alt_compare,
            o.map(Ordering::reverse),
            "Invalid alt comparison\n l: {:?}\n r: {:?}",
            l,
            r
        );

        //Test operators with faster implementations
        assert_eq!(
            matches!(compare, Some(Ordering::Less)),
            l < r,
            "Invalid (<):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Less) | Some(Ordering::Equal)),
            l <= r,
            "Invalid (<=):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Greater)),
            l > r,
            "Invalid (>):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Greater) | Some(Ordering::Equal)),
            l >= r,
            "Invalid (>=):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Less)),
            r < l,
            "Invalid alt (<):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Less) | Some(Ordering::Equal)),
            r <= l,
            "Invalid alt (<=):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Greater)),
            r > l,
            "Invalid alt (>):\n l: {:?}\n r: {:?}",
            l,
            r
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Greater) | Some(Ordering::Equal)),
            r >= l,
            "Invalid alt (>=):\n l: {:?}\n r: {:?}",
            l,
            r
        );
    }

    #[test]
    pub fn test_vclock_set() {
        let mut map = VSmallClockMap::default();
        let v1 = from_slice(&[3, 0, 1]);
        let v2 = from_slice(&[4, 2, 3]);
        let v3 = from_slice(&[4, 8, 3]);
        map.insert(VectorIdx(0), &v1);
        assert_eq!(map.get(VectorIdx(0)), Some(&v1));
        map.insert(VectorIdx(5), &v2);
        assert_eq!(map.get(VectorIdx(0)), Some(&v1));
        assert_eq!(map.get(VectorIdx(5)), Some(&v2));
        map.insert(VectorIdx(53), &v3);
        assert_eq!(map.get(VectorIdx(0)), Some(&v1));
        assert_eq!(map.get(VectorIdx(5)), Some(&v2));
        assert_eq!(map.get(VectorIdx(53)), Some(&v3));
        map.retain_index(VectorIdx(53));
        assert_eq!(map.get(VectorIdx(0)), None);
        assert_eq!(map.get(VectorIdx(5)), None);
        assert_eq!(map.get(VectorIdx(53)), Some(&v3));
        map.clear();
        assert_eq!(map.get(VectorIdx(0)), None);
        assert_eq!(map.get(VectorIdx(5)), None);
        assert_eq!(map.get(VectorIdx(53)), None);
        map.insert(VectorIdx(53), &v3);
        assert_eq!(map.get(VectorIdx(0)), None);
        assert_eq!(map.get(VectorIdx(5)), None);
        assert_eq!(map.get(VectorIdx(53)), Some(&v3));
    }
}
