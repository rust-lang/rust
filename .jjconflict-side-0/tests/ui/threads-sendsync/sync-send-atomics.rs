//@ check-pass


use std::sync::atomic::*;

trait SendSync: Send + Sync {}

impl SendSync for AtomicBool {}
impl SendSync for AtomicIsize {}
impl SendSync for AtomicUsize {}
impl<T> SendSync for AtomicPtr<T> {}

fn main() {}
