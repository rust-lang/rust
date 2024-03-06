//@ only-cdb
//@ compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx nonnull
// cdb-check:nonnull          : NonNull(0x[...]: 0xc) [Type: core::ptr::non_null::NonNull<u32>]
// cdb-check:    [<Raw View>]     [Type: core::ptr::non_null::NonNull<u32>]
// cdb-check:    0xc [Type: unsigned int]

// cdb-command: dx manuallydrop
// cdb-check:manuallydrop     : 12345 [Type: core::mem::manually_drop::ManuallyDrop<i32>]
// cdb-check:    [<Raw View>]     [Type: core::mem::manually_drop::ManuallyDrop<i32>]

// cdb-command: dx pin
// cdb-check:pin              : Pin(0x[...]: "this") [Type: core::pin::Pin<ref_mut$<alloc::string::String> >]
// cdb-check:    [<Raw View>]     [Type: core::pin::Pin<ref_mut$<alloc::string::String> >]
// cdb-check:    [len]            : 0x4 [Type: unsigned [...]]
// cdb-check:    [capacity]       : 0x4 [Type: unsigned [...]]
// cdb-check:    [chars]          : "this"

// cdb-command: dx unique
// cdb-check:unique           : Unique(0x[...]: (0x2a, 4321)) [Type: core::ptr::unique::Unique<tuple$<u64,i32> >]
// cdb-check:    [<Raw View>]     [Type: core::ptr::unique::Unique<tuple$<u64,i32> >]
// cdb-check:    [0]              : 0x2a [Type: unsigned __int64]
// cdb-check:    [1]              : 4321 [Type: int]

#![feature(ptr_internals)]

use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::ptr::{NonNull, Unique};

fn main() {
    let nonnull: NonNull<_> = (&12u32).into();

    let manuallydrop = ManuallyDrop::new(12345i32);

    let mut s = "this".to_string();
    let pin = Pin::new(&mut s);

    let unique: Unique<_> = (&mut (42u64, 4321i32)).into();

    zzz(); // #break
}

fn zzz() { }
