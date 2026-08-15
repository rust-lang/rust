//@ known-bug: #118784
//@ needs-rustc-debug-assertions

use std::collections::HashMap;

macro_rules! all_sync_send {
    ($ctor:expr, $($iter:expr),+) => ({
        $(
            let mut x = $ctor;
            is_sync(x.$iter());
            let mut y = $ctor;
            is_send(y.$iter());
        )+
    })
}

fn main() {
    all_sync_send!(HashMap, HashMap);
}
