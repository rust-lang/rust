//@ check-pass
#![deny(let_underscore_drop, let_underscore_lock)]

use std::sync::{Mutex, MutexGuard};

struct NontrivialDrop;

impl Drop for NontrivialDrop {
    fn drop(&mut self) {}
}

struct Guards<'a> {
    guard: MutexGuard<'a, i32>,
    drop: NontrivialDrop,
}

fn main() {
    let m = Mutex::new(0);

    let guard = m.lock().unwrap();
    let _ = guard;
    let _ = guard;
    drop(guard);

    let res = m.lock();
    let _ = res;
    drop(res);

    let mut d = NontrivialDrop;
    let _ = d;
    _ = d;
    let _ = *&d;
    let _ = *&mut d;
    drop(d);

    let pair = (m.lock().unwrap(), NontrivialDrop);
    let _ = pair.0;
    let (_, _) = pair;
    let (_, ref _d) = pair;
    drop(pair);

    let guards = Box::new(Guards { guard: m.lock().unwrap(), drop: NontrivialDrop });
    let _ = guards.guard;
    let _ = guards.drop;
    let _ = *guards;
    let Guards { guard: _, .. } = *guards;
    drop(guards);

    let nested = (0, Guards { guard: m.lock().unwrap(), drop: NontrivialDrop });
    let (_x, Guards { guard: _, .. }) = nested;
    drop(nested);

    let &_ = &m.lock().unwrap();
    let &_ = &(m.lock().unwrap(), NontrivialDrop).0;
    let &(_, _) = &(m.lock(), NontrivialDrop);

    let mut v = vec![NontrivialDrop];
    let _ = v[0];
    let _ = &mut v;
}
