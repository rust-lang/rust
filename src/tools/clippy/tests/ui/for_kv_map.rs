#![warn(clippy::for_kv_map)]
#![allow(clippy::used_underscore_binding)]

use std::collections::*;
use std::rc::Rc;

fn main() {
    let m: HashMap<u64, u64> = HashMap::new();
    for (_, v) in &m {
        //~^ ERROR: you seem to want to iterate on a map's values
        //~| NOTE: `-D clippy::for-kv-map` implied by `-D warnings`
        let _v = v;
    }

    let m: Rc<HashMap<u64, u64>> = Rc::new(HashMap::new());
    for (_, v) in &*m {
        //~^ ERROR: you seem to want to iterate on a map's values
        let _v = v;
        // Here the `*` is not actually necessary, but the test tests that we don't
        // suggest
        // `in *m.values()` as we used to
    }

    let mut m: HashMap<u64, u64> = HashMap::new();
    for (_, v) in &mut m {
        //~^ ERROR: you seem to want to iterate on a map's values
        let _v = v;
    }

    let m: &mut HashMap<u64, u64> = &mut HashMap::new();
    for (_, v) in &mut *m {
        //~^ ERROR: you seem to want to iterate on a map's values
        let _v = v;
    }

    let m: HashMap<u64, u64> = HashMap::new();
    let rm = &m;
    for (k, _value) in rm {
        //~^ ERROR: you seem to want to iterate on a map's keys
        let _k = k;
    }

    let m: HashMap<u64, u64> = HashMap::new();
    let rm = &m;
    'label: for (k, _value) in rm {
        //~^ ERROR: you seem to want to iterate on a map's keys
        let _k = k;
        if *k == 0u64 {
            break 'label;
        }
    }

    // The following should not produce warnings.

    let m: HashMap<u64, u64> = HashMap::new();
    // No error, _value is actually used
    for (k, _value) in &m {
        let _ = _value;
        let _k = k;
    }

    let m: HashMap<u64, String> = Default::default();
    for (_, v) in m {
        let _v = v;
    }
}
