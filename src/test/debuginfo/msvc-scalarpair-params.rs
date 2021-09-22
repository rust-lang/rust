// only-cdb
// compile-flags: -g

// cdb-command: g

// cdb-command: dx r1
// cdb-check:r1               : (0xa..0xc) [Type: core::ops::range::Range<u32>]
// cdb-command: dx r2
// cdb-check:r2               : (0x14..0x1e) [Type: core::ops::range::Range<u64>]

// cdb-command: g

// cdb-command: dx r1
// cdb-check:r1               : (0x9..0x64) [Type: core::ops::range::Range<u32>]
// cdb-command: dx r2
// cdb-check:r2               : (0xc..0x5a) [Type: core::ops::range::Range<u64>]

// cdb-command: g

// cdb-command: dx o1
// cdb-check:o1               : Some [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [variant]        : Some
// cdb-check:    [+0x004] __0              : 0x4d2 [Type: [...]]
// cdb-command: dx o2
// cdb-check:o2               : Some [Type: enum$<core::option::Option<u64> >]
// cdb-check:    [variant]        : Some
// cdb-check:    [+0x008] __0              : 0x162e [Type: unsigned __int64]

// cdb-command: g

// cdb-command: dx t1
// cdb-check:t1               : (0xa, 0x14) [Type: tuple$<u32,u32>]
// cdb-check:    [0]              : 0xa [Type: unsigned int]
// cdb-check:    [1]              : 0x14 [Type: unsigned int]
// cdb-command: dx t2
// cdb-check:t2               : (0x1e, 0x28) [Type: tuple$<u64,u64>]
// cdb-check:    [0]              : 0x1e [Type: unsigned __int64]
// cdb-check:    [1]              : 0x28 [Type: unsigned __int64]

// cdb-command: g

// cdb-command: dx s
// cdb-check:s                : "this is a static str" [Type: str]
// cdb-check:    [len]            : 0x14 [Type: unsigned [...]]
// cdb-check:    [chars]

// cdb-command: g

// cdb-command: dx s
// cdb-check:s                : { len=0x5 } [Type: slice$<u8>]
// cdb-check:    [len]            : 0x5 [Type: unsigned [...]]
// cdb-check:    [0]              : 0x1 [Type: unsigned char]
// cdb-check:    [1]              : 0x2 [Type: unsigned char]
// cdb-check:    [2]              : 0x3 [Type: unsigned char]
// cdb-check:    [3]              : 0x4 [Type: unsigned char]
// cdb-check:    [4]              : 0x5 [Type: unsigned char]

use std::ops::Range;

fn range(r1: Range<u32>, r2: Range<u64>) {
    zzz(); // #break
}

fn range_mut(mut r1: Range<u32>, mut r2: Range<u64>) {
    if r1.start == 9 {
        r1.end = 100;
    }

    if r2.start == 12 {
        r2.end = 90;
    }

    zzz(); // #break
}

fn option(o1: Option<u32>, o2: Option<u64>) {
    zzz(); // #break
}

fn tuple(t1: (u32, u32), t2: (u64, u64)) {
    zzz(); // #break
}

fn str(s: &str) {
    zzz(); // #break
}

fn slice(s: &[u8]) {
    zzz(); // #break
}

fn zzz() { }

fn main() {
    range(10..12, 20..30);
    range_mut(9..20, 12..80);
    option(Some(1234), Some(5678));
    tuple((10, 20), (30, 40));
    str("this is a static str");
    slice(&[1, 2, 3, 4, 5]);
}
