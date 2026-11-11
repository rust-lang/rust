#![warn(clippy::unnecessary_get_then_check)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

fn main() {
    let s: HashSet<String> = HashSet::new();
    let _ = s.get("a").is_some();
    //~^ unnecessary_get_then_check
    let _ = s.get("a").is_none();
    //~^ unnecessary_get_then_check

    let s: HashMap<String, ()> = HashMap::new();
    let _ = s.get("a").is_some();
    //~^ unnecessary_get_then_check
    let _ = s.get("a").is_none();
    //~^ unnecessary_get_then_check

    let s: BTreeSet<String> = BTreeSet::new();
    let _ = s.get("a").is_some();
    //~^ unnecessary_get_then_check
    let _ = s.get("a").is_none();
    //~^ unnecessary_get_then_check

    let s: BTreeMap<String, ()> = BTreeMap::new();
    let _ = s.get("a").is_some();
    //~^ unnecessary_get_then_check
    let _ = s.get("a").is_none();
    //~^ unnecessary_get_then_check

    // Import to check that the generic annotations are kept!
    let s: HashSet<String> = HashSet::new();
    let _ = s.get::<str>("a").is_some();
    //~^ unnecessary_get_then_check
    let _ = s.get::<str>("a").is_none();
    //~^ unnecessary_get_then_check
}
