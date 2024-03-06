// Check that we aren't using unsound specialization in slice comparisons.

//@ run-pass

use std::cell::Cell;
use std::cmp::Ordering;

struct Evil<'a, 'b>(Cell<(&'a [i32], &'b [i32])>);

impl PartialEq for Evil<'_, '_> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for Evil<'_, '_> {}

impl PartialOrd for Evil<'_, '_> {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl<'a> Ord for Evil<'a, 'a> {
    fn cmp(&self, _other: &Self) -> Ordering {
        let (a, b) = self.0.get();
        self.0.set((b, a));
        Ordering::Equal
    }
}

fn main() {
    let x = &[1, 2, 3, 4];
    let u = {
        let a = Box::new([7, 8, 9, 10]);
        let y = [Evil(Cell::new((x, &*a)))];
        let _ = &y[..] <= &y[..];
        let [Evil(c)] = y;
        c.get().0
    };
    assert_eq!(u, &[1, 2, 3, 4]);
}
