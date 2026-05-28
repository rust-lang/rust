//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc
//@ compile-flags: -C overflow-checks

use std::panic;

struct Lies(usize);

impl Iterator for Lies {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            self.0 -= 1;
            Some(self.0)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

fn main() {
    let r = panic::catch_unwind(|| {
        // This returns more items than its `size_hint` said was possible,
        // which `Filter::count` detects via `overflow-checks`.
        let _ = Lies(10).filter(|&x| x > 3).count();
    });
    assert!(r.is_err());
}
