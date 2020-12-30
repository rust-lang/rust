// CDB doesn't like how libstd.natvis casts to tuples before this version.
// https://github.com/rust-lang/rust/issues/76352#issuecomment-687640746
// min-cdb-version: 10.0.18362.1

// cdb-only
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx hash_set,d
// cdb-check:hash_set,d [...] : { len=15 } [Type: [...]::HashSet<u64, [...]>]
// cdb-check:    [len]            : 15 [Type: [...]]
// cdb-check:    [capacity]       : [...]
// cdb-check:    [[...]] [...]    : 0 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 1 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 2 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 3 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 4 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 5 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 6 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 7 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 8 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 9 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 10 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 11 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 12 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 13 [Type: u64]
// cdb-command: dx hash_set,d
// cdb-check:    [[...]] [...]    : 14 [Type: u64]

// cdb-command: dx hash_map,d
// cdb-check:hash_map,d [...] : { len=15 } [Type: [...]::HashMap<u64, u64, [...]>]
// cdb-check:    [len]            : 15 [Type: [...]]
// cdb-check:    [capacity]       : [...]
// cdb-check:    ["0x0"]          : 0 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x1"]          : 1 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x2"]          : 2 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x3"]          : 3 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x4"]          : 4 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x5"]          : 5 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x6"]          : 6 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x7"]          : 7 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x8"]          : 8 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0x9"]          : 9 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0xa"]          : 10 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0xb"]          : 11 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0xc"]          : 12 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0xd"]          : 13 [Type: unsigned __int64]
// cdb-command: dx hash_map,d
// cdb-check:    ["0xe"]          : 14 [Type: unsigned __int64]

#![allow(unused_variables)]
use std::collections::HashSet;
use std::collections::HashMap;


fn main() {
    // HashSet
    let mut hash_set = HashSet::new();
    for i in 0..15 {
        hash_set.insert(i as u64);
    }

    // HashMap
    let mut hash_map = HashMap::new();
    for i in 0..15 {
        hash_map.insert(i as u64, i as u64);
    }

    zzz(); // #break
}

fn zzz() { () }
