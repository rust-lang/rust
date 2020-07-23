// run-rustfix

#![allow(unused, clippy::suspicious_map)]

use std::collections::{BTreeSet, HashMap, HashSet};

#[warn(clippy::needless_collect)]
#[allow(unused_variables, clippy::iter_cloned_collect)]
fn main() {
    let sample = [1; 5];
    let len = sample.iter().collect::<Vec<_>>().len();
    if sample.iter().collect::<Vec<_>>().is_empty() {
        // Empty
    }
    sample.iter().cloned().collect::<Vec<_>>().contains(&1);
    sample.iter().map(|x| (x, x)).collect::<HashMap<_, _>>().len();
    // Notice the `HashSet`--this should not be linted
    sample.iter().collect::<HashSet<_>>().len();
    // Neither should this
    sample.iter().collect::<BTreeSet<_>>().len();
}
