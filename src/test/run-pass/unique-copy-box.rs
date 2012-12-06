extern mod std;
use sys::refcount;

fn main() unsafe {
    let i = ~@1;
    let j = ~@2;
    let rc1 = refcount(*i);
    let j = copy i;
    let rc2 = refcount(*i);
    error!("rc1: %u rc2: %u", rc1, rc2);
    assert rc1 + 1u == rc2;
}
