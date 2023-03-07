// Testing the display of range types in cdb.

// cdb-only
// min-cdb-version: 10.0.18317.1001
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx r1,d
// cdb-check:r1,d             : (3..5) [Type: core::ops::range::Range<i32>]
// cdb-check:    [<Raw View>]     [Type: core::ops::range::Range<i32>]

// cdb-command: dx r2,d
// cdb-check:r2,d             : (2..) [Type: core::ops::range::RangeFrom<i32>]
// cdb-check:    [<Raw View>]     [Type: core::ops::range::RangeFrom<i32>]

// cdb-command: dx r3,d
// cdb-check:r3,d             : (1..=4) [Type: core::ops::range::RangeInclusive<i32>]
// cdb-check:    [<Raw View>]     [Type: core::ops::range::RangeInclusive<i32>]

// cdb-command: dx r4,d
// cdb-check:r4,d             : (..10) [Type: core::ops::range::RangeTo<i32>]
// cdb-check:    [<Raw View>]     [Type: core::ops::range::RangeTo<i32>]

// cdb-command: dx r5,d
// cdb-check:r5,d             : (..=3) [Type: core::ops::range::RangeToInclusive<i32>]
// cdb-check:    [<Raw View>]     [Type: core::ops::range::RangeToInclusive<i32>]

// cdb-command: dx r6,d
// cdb-check:r6,d             [Type: core::ops::range::RangeFull]

#[allow(unused_variables)]

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeToInclusive};

fn main()
{
    let r1 = (3..5);
    let r2 = (2..);
    let r3 = (1..=4);
    let r4 = (..10);
    let r5 = (..=3);
    let r6 = (..);
    zzz(); // #break
}

fn zzz() { () }
