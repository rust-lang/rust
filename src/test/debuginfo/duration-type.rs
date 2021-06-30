// only-cdb
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx duration
// cdb-check:duration         : 5s 12ns [Type: core::time::Duration]
// cdb-check:    [<Raw View>]     [Type: core::time::Duration]
// cdb-check:    seconds          : 0x5 [Type: unsigned __int64]
// cdb-check:    nanoseconds      : 0xc [Type: unsigned int]

use std::time::Duration;

fn main() {
    let duration = Duration::new(5, 12);

    zzz(); // #break
}

fn zzz() { }
