use super::map::BTreeMap;
use super::set::BTreeSet;
use crate::vec::{IntoIter, Vec};

/// Specialization trait used for `BTreeMap::from_iter` and `BTreeSet::from_iter`.
pub(super) trait SpecFromIter<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<K, V, I> SpecFromIter<(K, V), I> for BTreeMap<K, V>
where
    K: Ord,
    I: Iterator<Item = (K, V)>,
{
    default fn spec_from_iter(iterator: I) -> Self {
        let mut inputs: Vec<_> = iterator.collect();

        if inputs.is_empty() {
            return BTreeMap::new();
        }

        // use stable sort to preserve the insertion order.
        inputs.sort_by(|a, b| a.0.cmp(&b.0));
        BTreeMap::bulk_build_from_sorted_iter(inputs)
    }
}

impl<K, V> SpecFromIter<(K, V), IntoIter<(K, V)>> for BTreeMap<K, V>
where
    K: Ord,
{
    fn spec_from_iter(mut iterator: IntoIter<(K, V)>) -> Self {
        if iterator.is_empty() {
            return BTreeMap::new();
        }

        // use stable sort to preserve the insertion order.
        iterator.as_mut_slice().sort_by(|a, b| a.0.cmp(&b.0));
        BTreeMap::bulk_build_from_sorted_iter(iterator)
    }
}

impl<T, I> SpecFromIter<T, I> for BTreeSet<T>
where
    T: Ord,
    I: Iterator<Item = T>,
{
    default fn spec_from_iter(iterator: I) -> Self {
        let mut inputs: Vec<_> = iterator.collect();

        if inputs.is_empty() {
            return BTreeSet::new();
        }

        // use stable sort to preserve the insertion order.
        inputs.sort();
        let iter = inputs.into_iter().map(|k| (k, ()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter);
        BTreeSet { map }
    }
}

impl<T> SpecFromIter<T, IntoIter<T>> for BTreeSet<T>
where
    T: Ord,
{
    fn spec_from_iter(mut iterator: IntoIter<T>) -> Self {
        if iterator.is_empty() {
            return BTreeSet::new();
        }

        // use stable sort to preserve the insertion order.
        iterator.as_mut_slice().sort();
        let iter = iterator.map(|k| (k, ()));
        let map = BTreeMap::bulk_build_from_sorted_iter(iter);
        BTreeSet { map }
    }
}
