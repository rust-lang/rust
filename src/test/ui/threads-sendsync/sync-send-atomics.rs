// run-pass

// pretty-expanded FIXME #23616

use std::sync::atomic::*;

trait SendSync: Send + Sync {}

impl SendSync for AtomicBool {}
impl SendSync for AtomicIsize {}
impl SendSync for AtomicUsize {}
impl<T> SendSync for AtomicPtr<T> {}

fn main() {}
