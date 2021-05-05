// run-rustfix

#![allow(unused, clippy::suspicious_map, clippy::iter_count)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
}
