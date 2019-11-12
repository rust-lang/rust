// build-pass

use std::sync::atomic::AtomicUsize;

static FOO: AtomicUsize = AtomicUsize::new(0);

const X: &AtomicUsize = &FOO;

fn main() {}
