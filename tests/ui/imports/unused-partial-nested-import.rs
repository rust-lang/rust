// run-rustfix
// check-pass

// Check that rustfix removes the braces around partial nested imports, if possible.

#![warn(unused_imports)]

use std::{collections::{HashMap, HashSet}, io::{Write, self, Read}}; //~ WARN unused imports: `HashSet`, `Write`, `self`
use std::collections::{BTreeMap, BTreeSet}; //~ WARN unused import: `BTreeSet`

fn main() {
    let _: HashMap<(), ()> = HashMap::new();
    let _: BTreeMap<(), ()> = BTreeMap::new();
    let _ = (&*vec![1, 2, 3]).read(&mut [0, 0, 0]);
}
