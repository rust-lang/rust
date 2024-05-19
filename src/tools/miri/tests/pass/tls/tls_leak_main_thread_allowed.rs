//@ignore-target-windows: Windows uses a different mechanism for `thread_local!`
#![feature(thread_local)]

use std::cell::Cell;

// Thread-local variables in the main thread are basically like `static` (they live
// as long as the program does), so make sure we treat them the same for leak purposes.
//
// The test covers both TLS statics and the TLS macro.
pub fn main() {
    thread_local! {
        static TLS_KEY: Cell<Option<&'static i32>> = Cell::new(None);
    }

    TLS_KEY.with(|cell| {
        cell.set(Some(Box::leak(Box::new(123))));
    });

    #[thread_local]
    static TLS: Cell<Option<&'static i32>> = Cell::new(None);

    TLS.set(Some(Box::leak(Box::new(123))));
}
