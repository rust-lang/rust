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
    //~^ iter_kv_map
    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().iter().map(|(_, val)| val).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.iter().map(|(key, _)| key).filter(|x| *x % 2 == 0).count();
    //~^ iter_kv_map

    // Don't lint
    let _ = map.iter().filter(|(_, val)| *val % 2 == 0).map(|(key, _)| key).count();
    let _ = map.iter().map(get_key).collect::<Vec<_>>();

    // Linting the following could be an improvement to the lint
    // map.iter().filter_map(|(_, val)| (val % 2 == 0).then(val * 17)).count();

    // Lint
    let _ = map.iter().map(|(key, _value)| key * 9).count();
    //~^ iter_kv_map
    let _ = map.iter().map(|(_key, value)| value * 17).count();
    //~^ iter_kv_map

    // Preserve the ref in the fix.
    let _ = map.clone().into_iter().map(|(_, ref val)| ref_acceptor(val)).count();
    //~^ iter_kv_map

    // Preserve the mut in the fix.
    let _ = map
        //~^ iter_kv_map
        .clone()
        .into_iter()
        .map(|(_, mut val)| {
            val += 2;
            val
        })
        .count();

    // Don't let a mut interfere.
    let _ = map.clone().into_iter().map(|(_, mut val)| val).count();
    //~^ iter_kv_map

    let map: BTreeMap<u32, u32> = BTreeMap::new();

    let _ = map.iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().iter().map(|(_, val)| val).collect::<Vec<_>>();
    //~^ iter_kv_map
    let _ = map.iter().map(|(key, _)| key).filter(|x| *x % 2 == 0).count();
    //~^ iter_kv_map

    // Don't lint
    let _ = map.iter().filter(|(_, val)| *val % 2 == 0).map(|(key, _)| key).count();
    let _ = map.iter().map(get_key).collect::<Vec<_>>();

    // Linting the following could be an improvement to the lint
    // map.iter().filter_map(|(_, val)| (val % 2 == 0).then(val * 17)).count();

    // Lint
    let _ = map.iter().map(|(key, _value)| key * 9).count();
    //~^ iter_kv_map
    let _ = map.iter().map(|(_key, value)| value * 17).count();
    //~^ iter_kv_map

    // Preserve the ref in the fix.
    let _ = map.clone().into_iter().map(|(_, ref val)| ref_acceptor(val)).count();
    //~^ iter_kv_map

    // Preserve the mut in the fix.
    let _ = map
        //~^ iter_kv_map
        .clone()
        .into_iter()
        .map(|(_, mut val)| {
            val += 2;
            val
        })
        .count();

    // Don't let a mut interfere.
    let _ = map.clone().into_iter().map(|(_, mut val)| val).count();
    //~^ iter_kv_map
}

#[clippy::msrv = "1.53"]
fn msrv_1_53() {
    let map: HashMap<u32, u32> = HashMap::new();

    // Don't lint because into_iter is not supported
    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();

    // Lint
    let _ = map.iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();
    //~^ iter_kv_map
}

#[clippy::msrv = "1.54"]
fn msrv_1_54() {
    // Lint all
    let map: HashMap<u32, u32> = HashMap::new();

    let _ = map.clone().into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(key, _)| key + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(_, val)| val).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.clone().into_iter().map(|(_, val)| val + 2).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.iter().map(|(key, _)| key).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    //~^ iter_kv_map

    let _ = map.iter().map(|(_, v)| v + 2).collect::<Vec<_>>();
    //~^ iter_kv_map
}
