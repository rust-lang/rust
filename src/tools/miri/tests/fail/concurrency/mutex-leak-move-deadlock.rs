//@error-in-other-file: deadlock
//@normalize-stderr-test: "src/sys/.*\.rs" -> "$$FILE"
//@normalize-stderr-test: "LL \| .*" -> "LL | $$CODE"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n *= note:.*" -> ""
use std::mem;
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(0);
    mem::forget(m.lock());
    // Move the lock while it is "held" (really: leaked)
    let m2 = m;
    // Now try to acquire the lock again.
    let _guard = m2.lock();
}
