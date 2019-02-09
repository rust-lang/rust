// compile-flags: --crate-type=lib

// pp-exact

fn f(v: &[isize]) -> isize {
    let mut n = 0;
    for e in v {
        n = *e; // This comment once triggered a pretty-printer bug.
    }
    n
}
