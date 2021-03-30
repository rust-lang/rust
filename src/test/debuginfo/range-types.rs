// Testing the display of range types in cdb.

// cdb-only
// min-cdb-version: 10.0.18317.1001
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx r1,d
// cdb-check:r1,d             [Type: core::ops::range::Range<i32>]
// cdb-check:    [...] start            : 3 [Type: int]
// cdb-check:    [...] end              : 5 [Type: int]

// cdb-command: dx r2,d
// cdb-check:r2,d             [Type: core::ops::range::RangeFrom<i32>]
// cdb-check:    [...] start            : 2 [Type: int]

// cdb-command: dx r3,d
// cdb-check:r3,d             [Type: core::ops::range::RangeInclusive<i32>]
// cdb-check:    [...] start            : 1 [Type: int]
// cdb-check:    [...] end              : 4 [Type: int]
// cdb-check:    [...] exhausted        : false [Type: bool]

// cdb-command: dx r4,d
// cdb-check:r4,d             [Type: core::ops::range::RangeToInclusive<i32>]
// cdb-check:    [...] end              : 3 [Type: int]

// cdb-command: dx r5,d
// cdb-check:r5,d             [Type: core::ops::range::RangeFull]

#[allow(unused_variables)]

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeToInclusive};

fn main()
{
    let r1 = Range{start: 3, end: 5};
    let r2 = RangeFrom{start: 2};
    let r3 = RangeInclusive::new(1, 4);
    let r4 = RangeToInclusive{end: 3};
    let r5 = RangeFull{};
    zzz(); // #break
}

fn zzz() { () }
