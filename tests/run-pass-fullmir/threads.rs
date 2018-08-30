// Just instantiate some data structures to make sure we got all their foreign items covered.
// Requires full MIR on Windows.

use std::sync;

fn main() {
    let m = sync::Mutex::new(0);
    let _ = m.lock();
    drop(m);

    let rw = sync::RwLock::new(0);
    let _ = rw.read();
    let _ = rw.write();
    drop(rw);
}
