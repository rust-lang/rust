use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Index, Shr};

use rustc_index::Idx;
use rustc_span::{DUMMY_SP, Span, SpanData};
use smallvec::SmallVec;

use super::data_race::NaReadType;
use crate::helpers::ToUsize;

/// A vector clock index, this is associated with a thread id
/// but in some cases one vector index may be shared with
/// multiple thread ids if it's safe to do so.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub(super) struct VectorIdx(u32);

impl VectorIdx {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.0
    }
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

/// The size of the vector clock to store inline.
/// Clock vectors larger than this will be stored on the heap.
const SMALL_VECTOR: usize = 4;

/// The time-stamps recorded in the data-race detector consist of both
/// a 32-bit unsigned integer which is the actual timestamp, and a `Span`
/// so that diagnostics can report what code was responsible for an operation.
#[derive(Clone, Copy, Debug)]
pub(super) struct VTimestamp {
    /// The lowest bit indicates read type, the rest is the time.
    /// `1` indicates a retag read, `0` a regular read.
    time_and_read_type: u32,
    pub span: Span,
}

impl VTimestamp {
    pub const ZERO: VTimestamp = VTimestamp::new(0, NaReadType::Read, DUMMY_SP);

    #[inline]
    const fn encode_time_and_read_type(time: u32, read_type: NaReadType) -> u32 {
        let read_type_bit = match read_type {
            NaReadType::Read => 0,
            NaReadType::Retag => 1,
        };
        // Put the `read_type` in the lowest bit and `time` in the rest
        read_type_bit | time.checked_mul(2).expect("Vector clock overflow")
    }

    #[inline]
    const fn new(time: u32, read_type: NaReadType, span: Span) -> Self {
        Self { time_and_read_type: Self::encode_time_and_read_type(time, read_type), span }
    }

    #[inline]
    fn time(&self) -> u32 {
        self.time_and_read_type.shr(1)
    }

    #[inline]
    fn set_time(&mut self, time: u32) {
        self.time_and_read_type = Self::encode_time_and_read_type(time, self.read_type());
    }

    #[inline]
    pub(super) fn read_type(&self) -> NaReadType {
        if self.time_and_read_type & 1 == 0 { NaReadType::Read } else { NaReadType::Retag }
    }

    #[inline]
    pub(super) fn set_read_type(&mut self, read_type: NaReadType) {
        self.time_and_read_type = Self::encode_time_and_read_type(self.time(), read_type);
    }

    #[inline]
    pub(super) fn span_data(&self) -> SpanData {
        self.span.data()
    }
}

impl PartialEq for VTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.time() == other.time()
    }
}

impl Eq for VTimestamp {}

impl PartialOrd for VTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VTimestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time().cmp(&other.time())
    }
}

/// A vector clock for detecting data-races, this is conceptually
/// a map from a vector index (and thus a thread id) to a timestamp.
/// The compare operations require that the invariant that the last
/// element in the internal timestamp slice must not be a 0, hence
/// all zero vector clocks are always represented by the empty slice;
/// and allows for the implementation of compare operations to short
/// circuit the calculation and return the correct result faster,
/// also this means that there is only one unique valid length
/// for each set of vector clock values and hence the PartialEq
/// and Eq derivations are correct.
///
/// This means we cannot represent a clock where the last entry is a timestamp-0 read that occurs
/// because of a retag. That's fine, all it does is risk wrong diagnostics in a extreme corner case.
#[derive(PartialEq, Eq, Default, Debug)]
pub struct VClock(SmallVec<[VTimestamp; SMALL_VECTOR]>);

impl VClock {
    /// Create a new vector clock containing all zeros except
    /// for a value at the given index
    pub(super) fn new_with_index(index: VectorIdx, timestamp: VTimestamp) -> VClock {
        if timestamp.time() == 0 {
            return VClock::default();
        }
        let len = index.index() + 1;
        let mut vec = smallvec::smallvec![VTimestamp::ZERO; len];
        vec[index.index()] = timestamp;
        VClock(vec)
    }

    /// Load the internal timestamp slice in the vector clock
    #[inline]
    pub(super) fn as_slice(&self) -> &[VTimestamp] {
        debug_assert!(self.0.last().is_none_or(|t| t.time() != 0));
        self.0.as_slice()
    }

    #[inline]
    pub(super) fn index_mut(&mut self, index: VectorIdx) -> &mut VTimestamp {
        self.0.as_mut_slice().get_mut(index.to_u32().to_usize()).unwrap()
    }

    /// Get a mutable slice to the internal vector with minimum `min_len`
    /// elements. To preserve invariants, the caller must modify
    /// the `min_len`-1 nth element to a non-zero value
    #[inline]
    fn get_mut_with_min_len(&mut self, min_len: usize) -> &mut [VTimestamp] {
        if self.0.len() < min_len {
            self.0.resize(min_len, VTimestamp::ZERO);
        }
        assert!(self.0.len() >= min_len);
        self.0.as_mut_slice()
    }

    /// Increment the vector clock at a known index
    /// this will panic if the vector index overflows
    #[inline]
    pub(super) fn increment_index(&mut self, idx: VectorIdx, current_span: Span) {
        let idx = idx.index();
        let mut_slice = self.get_mut_with_min_len(idx + 1);
        let idx_ref = &mut mut_slice[idx];
        idx_ref.set_time(idx_ref.time().checked_add(1).expect("Vector clock overflow"));
        if !current_span.is_dummy() {
            idx_ref.span = current_span;
        }
    }

    // Join the two vector clocks together, this
    // sets each vector element to the maximum value
    // of that element in either of the two source elements.
    pub fn join(&mut self, other: &Self) {
        let rhs_slice = other.as_slice();
        let lhs_slice = self.get_mut_with_min_len(rhs_slice.len());
        for (l, &r) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            let l_span = l.span;
            let r_span = r.span;
            *l = r.max(*l);
            l.span = l.span.substitute_dummy(r_span).substitute_dummy(l_span);
        }
    }

    /// Set the element at the current index of the vector. May only increase elements.
    pub(super) fn set_at_index(&mut self, other: &Self, idx: VectorIdx) {
        let new_timestamp = other[idx];
        // Setting to 0 is different, since the last element cannot be 0.
        if new_timestamp.time() == 0 {
            if idx.index() >= self.0.len() {
                // This index does not even exist yet in our clock. Just do nothing.
                return;
            }
            // This changes an existing element. Since it can only increase, that
            // can never make the last element 0.
        }

        let mut_slice = self.get_mut_with_min_len(idx.index() + 1);
        let mut_timestamp = &mut mut_slice[idx.index()];

        let prev_span = mut_timestamp.span;

        assert!(*mut_timestamp <= new_timestamp, "set_at_index: may only increase the timestamp");
        *mut_timestamp = new_timestamp;

        let span = &mut mut_timestamp.span;
        *span = span.substitute_dummy(prev_span);
    }

    /// Set the vector to the all-zero vector
    #[inline]
    pub(super) fn set_zero_vector(&mut self) {
        self.0.clear();
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
            Ordering::Less =>
                match order {
                    Ordering::Less | Ordering::Equal => Some(Ordering::Less),
                    Ordering::Greater => None,
                },
            // Left has at least 1 element > than the implicit 0,
            // so the only valid values are Ordering::Greater or None.
            Ordering::Greater =>
                match order {
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
        self.as_slice().get(index.to_u32().to_usize()).unwrap_or(&VTimestamp::ZERO)
    }
}

/// Test vector clock ordering operations
///  data-race detection is tested in the external
///  test suite
#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rustc_span::DUMMY_SP;

    use super::{VClock, VTimestamp, VectorIdx};
    use crate::concurrency::data_race::NaReadType;

    #[test]
    fn test_equal() {
        let mut c1 = VClock::default();
        let mut c2 = VClock::default();
        assert_eq!(c1, c2);
        c1.increment_index(VectorIdx(5), DUMMY_SP);
        assert_ne!(c1, c2);
        c2.increment_index(VectorIdx(53), DUMMY_SP);
        assert_ne!(c1, c2);
        c1.increment_index(VectorIdx(53), DUMMY_SP);
        assert_ne!(c1, c2);
        c2.increment_index(VectorIdx(5), DUMMY_SP);
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

    fn from_slice(mut slice: &[u32]) -> VClock {
        while let Some(0) = slice.last() {
            slice = &slice[..slice.len() - 1]
        }
        VClock(
            slice
                .iter()
                .copied()
                .map(|time| VTimestamp::new(time, NaReadType::Read, DUMMY_SP))
                .collect(),
        )
    }

    fn assert_order(l: &[u32], r: &[u32], o: Option<Ordering>) {
        let l = from_slice(l);
        let r = from_slice(r);

        //Test partial_cmp
        let compare = l.partial_cmp(&r);
        assert_eq!(compare, o, "Invalid comparison\n l: {l:?}\n r: {r:?}");
        let alt_compare = r.partial_cmp(&l);
        assert_eq!(
            alt_compare,
            o.map(Ordering::reverse),
            "Invalid alt comparison\n l: {l:?}\n r: {r:?}"
        );

        //Test operators with faster implementations
        assert_eq!(
            matches!(compare, Some(Ordering::Less)),
            l < r,
            "Invalid (<):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Less) | Some(Ordering::Equal)),
            l <= r,
            "Invalid (<=):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Greater)),
            l > r,
            "Invalid (>):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(compare, Some(Ordering::Greater) | Some(Ordering::Equal)),
            l >= r,
            "Invalid (>=):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Less)),
            r < l,
            "Invalid alt (<):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Less) | Some(Ordering::Equal)),
            r <= l,
            "Invalid alt (<=):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Greater)),
            r > l,
            "Invalid alt (>):\n l: {l:?}\n r: {r:?}"
        );
        assert_eq!(
            matches!(alt_compare, Some(Ordering::Greater) | Some(Ordering::Equal)),
            r >= l,
            "Invalid alt (>=):\n l: {l:?}\n r: {r:?}"
        );
    }

    #[test]
    fn set_index_to_0() {
        let mut clock1 = from_slice(&[0, 1, 2, 3]);
        let clock2 = from_slice(&[0, 2, 3, 4, 0, 5]);
        // Naively, this would extend clock1 with a new index and set it to 0, making
        // the last index 0. Make sure that does not happen.
        clock1.set_at_index(&clock2, VectorIdx(4));
        // This must not have made the last element 0.
        assert!(clock1.0.last().unwrap().time() != 0);
    }
}
