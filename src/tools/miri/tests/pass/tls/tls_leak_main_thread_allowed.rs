#![feature(thread_local, cfg_target_thread_local)]

use std::cell::Cell;

// Thread-local variables in the main thread are basically like `static` (they live
// as long as the program does), so make sure we treat them the same for leak purposes.
//
// The test covers both TLS statics and the TLS macro.
pub fn main() {
    #[thread_local]
    static TLS: Cell<Option<&'static i32>> = Cell::new(None);

    TLS.set(Some(Box::leak(Box::new(123))));

    // We can only ignore leaks on targets that use `#[thread_local]` statics to implement
    // `thread_local!`. Ignore the test on targets that don't.
    if cfg!(target_thread_local) {
        thread_local! {
            static TLS_KEY: Cell<Option<&'static i32>> = Cell::new(None);
        }

        TLS_KEY.with(|cell| {
            cell.set(Some(Box::leak(Box::new(123))));
        });
    }
}
