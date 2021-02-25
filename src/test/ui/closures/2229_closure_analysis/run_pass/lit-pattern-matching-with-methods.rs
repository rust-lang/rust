//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
#![warn(unused)]
#![feature(rustc_attrs)]
#![feature(btree_drain_filter)]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    let mut map = BTreeMap::new();
    map.insert("a", ());
    map.insert("b", ());
    map.insert("c", ());

    {
        let mut it = map.drain_filter(|_, _| true);
        catch_unwind(AssertUnwindSafe(|| while it.next().is_some() {})).unwrap_err();
        let result = catch_unwind(AssertUnwindSafe(|| it.next()));
        assert!(matches!(result, Ok(None)));
    }

    {
        let mut it = map.drain_filter(|_, _| true);
        catch_unwind(AssertUnwindSafe(|| while let Some(_) = it.next() {})).unwrap_err();
        let result = catch_unwind(AssertUnwindSafe(|| it.next()));
        assert!(matches!(result, Ok(None)));
    }

}
