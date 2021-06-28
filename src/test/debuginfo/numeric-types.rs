// only-cdb
// compile-flags:-g

// Tests the visualizations for `NonZero{I,U}{8,16,32,64,128,size}` and `Wrapping<T>` in
// `libcore.natvis`.

// === CDB TESTS ==================================================================================
// cdb-command: g

// cdb-command: dx nz_i8
// cdb-check:nz_i8            : 11 [Type: core::num::nonzero::NonZeroI8]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI8]

// cdb-command: dx nz_i16
// cdb-check:nz_i16           : 22 [Type: core::num::nonzero::NonZeroI16]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI16]

// cdb-command: dx nz_i32
// cdb-check:nz_i32           : 33 [Type: core::num::nonzero::NonZeroI32]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI32]

// cdb-command: dx nz_i64
// cdb-check:nz_i64           : 44 [Type: core::num::nonzero::NonZeroI64]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI64]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx nz_i128
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroI128]

// cdb-command: dx nz_isize
// cdb-check:nz_isize         : 66 [Type: core::num::nonzero::NonZeroIsize]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroIsize]

// cdb-command: dx nz_u8
// cdb-check:nz_u8            : 0x4d [Type: core::num::nonzero::NonZeroU8]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU8]

// cdb-command: dx nz_u16
// cdb-check:nz_u16           : 0x58 [Type: core::num::nonzero::NonZeroU16]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU16]

// cdb-command: dx nz_u32
// cdb-check:nz_u32           : 0x63 [Type: core::num::nonzero::NonZeroU32]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU32]

// cdb-command: dx nz_u64
// cdb-check:nz_u64           : 0x64 [Type: core::num::nonzero::NonZeroU64]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU64]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx nz_u128
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroU128]

// cdb-command: dx nz_usize
// cdb-check:nz_usize         : 0x7a [Type: core::num::nonzero::NonZeroUsize]
// cdb-check:    [<Raw View>]     [Type: core::num::nonzero::NonZeroUsize]

// cdb-command: dx w_i8
// cdb-check:w_i8             : 10 [Type: core::num::wrapping::Wrapping<i8>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i8>]

// cdb-command: dx w_i16
// cdb-check:w_i16            : 20 [Type: core::num::wrapping::Wrapping<i16>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i16>]

// cdb-command: dx w_i32
// cdb-check:w_i32            : 30 [Type: core::num::wrapping::Wrapping<i32>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i32>]

// cdb-command: dx w_i64
// cdb-check:w_i64            : 40 [Type: core::num::wrapping::Wrapping<i64>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i64>]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx w_i128
// cdb-check:w_i128           [Type: core::num::wrapping::Wrapping<i128>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<i128>]

// cdb-command: dx w_isize
// cdb-check:w_isize          : 60 [Type: core::num::wrapping::Wrapping<isize>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<isize>]

// cdb-command: dx w_u8
// cdb-check:w_u8             : 0x46 [Type: core::num::wrapping::Wrapping<u8>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u8>]

// cdb-command: dx w_u16
// cdb-check:w_u16            : 0x50 [Type: core::num::wrapping::Wrapping<u16>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u16>]

// cdb-command: dx w_u32
// cdb-check:w_u32            : 0x5a [Type: core::num::wrapping::Wrapping<u32>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u32>]

// cdb-command: dx w_u64
// cdb-check:w_u64            : 0x64 [Type: core::num::wrapping::Wrapping<u64>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u64>]

// 128-bit integers don't seem to work in CDB
// cdb-command: dx w_u128
// cdb-check:w_u128           [Type: core::num::wrapping::Wrapping<u128>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<u128>]

// cdb-command: dx w_usize
// cdb-check:w_usize          : 0x78 [Type: core::num::wrapping::Wrapping<usize>]
// cdb-check:    [<Raw View>]     [Type: core::num::wrapping::Wrapping<usize>]

use std::num::*;

fn main() {
    let nz_i8 = NonZeroI8::new(11).unwrap();
    let nz_i16 = NonZeroI16::new(22).unwrap();
    let nz_i32 = NonZeroI32::new(33).unwrap();
    let nz_i64 = NonZeroI64::new(44).unwrap();
    let nz_i128 = NonZeroI128::new(55).unwrap();
    let nz_isize = NonZeroIsize::new(66).unwrap();

    let nz_u8 = NonZeroU8::new(77).unwrap();
    let nz_u16 = NonZeroU16::new(88).unwrap();
    let nz_u32 = NonZeroU32::new(99).unwrap();
    let nz_u64 = NonZeroU64::new(100).unwrap();
    let nz_u128 = NonZeroU128::new(111).unwrap();
    let nz_usize = NonZeroUsize::new(122).unwrap();

    let w_i8 = Wrapping(10i8);
    let w_i16 = Wrapping(20i16);
    let w_i32 = Wrapping(30i32);
    let w_i64 = Wrapping(40i64);
    let w_i128 = Wrapping(50i128);
    let w_isize = Wrapping(60isize);

    let w_u8 = Wrapping(70u8);
    let w_u16 = Wrapping(80u16);
    let w_u32 = Wrapping(90u32);
    let w_u64 = Wrapping(100u64);
    let w_u128 = Wrapping(110u128);
    let w_usize = Wrapping(120usize);

    zzz(); // #break
}

fn zzz() { }
