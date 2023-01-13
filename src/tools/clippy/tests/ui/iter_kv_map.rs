// run-rustfix

#![warn(clippy::iter_kv_map)]
#![allow(unused_mut, clippy::redundant_clone, clippy::suspicious_map, clippy::map_identity)]

use std::collections::{BTreeMap, HashMap};

fn main() {
    let get_key = |(key, _val)| key;
    fn ref_acceptor(v: &u32) -> u32 {
        *v
    }

    let map: HashMap<u32, u32> = HashMap::new();

    let _ = map.iter().map(|(key, _)| key).collect::<Vec<_>>();
    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();

    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();

    let _ = map.clone().iter().map(|(_, val)| val).collect::<Vec<_>>();
    let _ = map.iter().map(|(key, _)| key).filter(|x| *x % 2 == 0).count();

    // Don't lint
    let _ = map.iter().filter(|(_, val)| *val % 2 == 0).map(|(key, _)| key).count();
    let _ = map.iter().map(get_key).collect::<Vec<_>>();

    // Linting the following could be an improvement to the lint
    // map.iter().filter_map(|(_, val)| (val % 2 == 0).then(val * 17)).count();

    // Lint
    let _ = map.iter().map(|(key, _value)| key * 9).count();
    let _ = map.iter().map(|(_key, value)| value * 17).count();

    // Preserve the ref in the fix.
    let _ = map.clone().into_iter().map(|(_, ref val)| ref_acceptor(val)).count();

    // Preserve the mut in the fix.
    let _ = map
        .clone()
        .into_iter()
        .map(|(_, mut val)| {
            val += 2;
            val
        })
        .count();

    // Don't let a mut interfere.
    let _ = map.clone().into_iter().map(|(_, mut val)| val).count();

    let map: BTreeMap<u32, u32> = BTreeMap::new();

    let _ = map.iter().map(|(key, _)| key).collect::<Vec<_>>();
    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();

    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();

    let _ = map.clone().iter().map(|(_, val)| val).collect::<Vec<_>>();
    let _ = map.iter().map(|(key, _)| key).filter(|x| *x % 2 == 0).count();

    // Don't lint
    let _ = map.iter().filter(|(_, val)| *val % 2 == 0).map(|(key, _)| key).count();
    let _ = map.iter().map(get_key).collect::<Vec<_>>();

    // Linting the following could be an improvement to the lint
    // map.iter().filter_map(|(_, val)| (val % 2 == 0).then(val * 17)).count();

    // Lint
    let _ = map.iter().map(|(key, _value)| key * 9).count();
    let _ = map.iter().map(|(_key, value)| value * 17).count();

    // Preserve the ref in the fix.
    let _ = map.clone().into_iter().map(|(_, ref val)| ref_acceptor(val)).count();

    // Preserve the mut in the fix.
    let _ = map
        .clone()
        .into_iter()
        .map(|(_, mut val)| {
            val += 2;
            val
        })
        .count();

    // Don't let a mut interfere.
    let _ = map.clone().into_iter().map(|(_, mut val)| val).count();
}
