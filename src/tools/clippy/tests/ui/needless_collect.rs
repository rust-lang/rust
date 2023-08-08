//@run-rustfix

#![allow(unused, clippy::needless_if, clippy::suspicious_map, clippy::iter_count)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList};

#[warn(clippy::needless_collect)]
#[allow(unused_variables, clippy::iter_cloned_collect, clippy::iter_next_slice)]
fn main() {
    let sample = [1; 5];
    let len = sample.iter().collect::<Vec<_>>().len();
    if sample.iter().collect::<Vec<_>>().is_empty() {
        // Empty
    }
    sample.iter().cloned().collect::<Vec<_>>().contains(&1);
    // #7164 HashMap's and BTreeMap's `len` usage should not be linted
    sample.iter().map(|x| (x, x)).collect::<HashMap<_, _>>().len();
    sample.iter().map(|x| (x, x)).collect::<BTreeMap<_, _>>().len();

    sample.iter().map(|x| (x, x)).collect::<HashMap<_, _>>().is_empty();
    sample.iter().map(|x| (x, x)).collect::<BTreeMap<_, _>>().is_empty();

    // Notice the `HashSet`--this should not be linted
    sample.iter().collect::<HashSet<_>>().len();
    // Neither should this
    sample.iter().collect::<BTreeSet<_>>().len();

    sample.iter().collect::<LinkedList<_>>().len();
    sample.iter().collect::<LinkedList<_>>().is_empty();
    sample.iter().cloned().collect::<LinkedList<_>>().contains(&1);
    sample.iter().collect::<LinkedList<_>>().contains(&&1);

    // `BinaryHeap` doesn't have `contains` method
    sample.iter().collect::<BinaryHeap<_>>().len();
    sample.iter().collect::<BinaryHeap<_>>().is_empty();

    // Don't lint string from str
    let _ = ["", ""].into_iter().collect::<String>().is_empty();

    let _ = sample.iter().collect::<HashSet<_>>().is_empty();
    let _ = sample.iter().collect::<HashSet<_>>().contains(&&0);

    struct VecWrapper<T>(Vec<T>);
    impl<T> core::ops::Deref for VecWrapper<T> {
        type Target = Vec<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T> IntoIterator for VecWrapper<T> {
        type IntoIter = <Vec<T> as IntoIterator>::IntoIter;
        type Item = <Vec<T> as IntoIterator>::Item;
        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }
    impl<T> FromIterator<T> for VecWrapper<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self(Vec::from_iter(iter))
        }
    }

    let _ = sample.iter().collect::<VecWrapper<_>>().is_empty();
    let _ = sample.iter().collect::<VecWrapper<_>>().contains(&&0);

    #[allow(clippy::double_parens)]
    {
        Vec::<u8>::new().extend((0..10).collect::<Vec<_>>());
        foo((0..10).collect::<Vec<_>>());
        bar((0..10).collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
        baz((0..10), (), ('a'..='z').collect::<Vec<_>>())
    }

    let values = [1, 2, 3, 4];
    let mut out = vec![];
    values.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>();
    let _y = values.iter().cloned().map(|x| out.push(x)).collect::<Vec<_>>(); // this is fine
}

fn foo(_: impl IntoIterator<Item = usize>) {}
fn bar<I: IntoIterator<Item = usize>>(_: Vec<usize>, _: I) {}
fn baz<I: IntoIterator<Item = usize>>(_: I, _: (), _: impl IntoIterator<Item = char>) {}
