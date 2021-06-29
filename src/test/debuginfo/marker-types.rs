// only-cdb
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx nonnull
// cdb-check:nonnull          : NonNull(0x[...]: 0xc) [Type: core::ptr::non_null::NonNull<u32>]
// cdb-check:    [<Raw View>]     [Type: core::ptr::non_null::NonNull<u32>]
// cdb-checK:    0xc [Type: unsigned int]

use std::ptr::NonNull;

fn main() {
    let nonnull: NonNull<_> = (&12u32).into();

    zzz(); // #break
}

fn zzz() { }
