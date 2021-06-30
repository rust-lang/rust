// only-cdb
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx nonnull
// cdb-check:nonnull          : NonNull(0x[...]: 0xc) [Type: core::ptr::non_null::NonNull<u32>]
// cdb-check:    [<Raw View>]     [Type: core::ptr::non_null::NonNull<u32>]
// cdb-checK:    0xc [Type: unsigned int]

// cdb-command: dx manuallydrop
// cdb-check:manuallydrop     : 12345 [Type: core::mem::manually_drop::ManuallyDrop<i32>]
// cdb-check:    [<Raw View>]     [Type: core::mem::manually_drop::ManuallyDrop<i32>]

// cdb-command: dx pin
// cdb-check:pin              : Pin(0x[...]: "this") [Type: core::pin::Pin<ref_mut$<alloc::string::String> >]
// cdb-check:    [<Raw View>]     [Type: core::pin::Pin<ref_mut$<alloc::string::String> >]
// cdb-check:    [len]            : 0x4 [Type: unsigned __int64]
// cdb-check:    [capacity]       : 0x4 [Type: unsigned __int64]
// cdb-check:    [chars]

use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::ptr::NonNull;

fn main() {
    let nonnull: NonNull<_> = (&12u32).into();

    let manuallydrop = ManuallyDrop::new(12345i32);

    let mut s = "this".to_string();
    let pin = Pin::new(&mut s);

    zzz(); // #break
}

fn zzz() { }
