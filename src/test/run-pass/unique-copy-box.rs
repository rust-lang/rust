use std;
import sys::refcount;

fn main() unsafe {
    let i = ~@1;
    let j = ~@2;
    let rc1 = refcount(*i);
    let j = i;
    let rc2 = refcount(*i);
    log_err #fmt("rc1: %u rc2: %u", rc1, rc2);
    assert rc1 + 1u == rc2;
}