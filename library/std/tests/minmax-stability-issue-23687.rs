use std::cmp::{self, Ordering};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Foo {
    n: u8,
    name: &'static str,
}

impl PartialOrd for Foo {
    fn partial_cmp(&self, other: &Foo) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Foo {
    fn cmp(&self, other: &Foo) -> Ordering {
        self.n.cmp(&other.n)
    }
}

#[test]
fn minmax_stability() {
    let a = Foo { n: 4, name: "a" };
    let b = Foo { n: 4, name: "b" };
    let c = Foo { n: 8, name: "c" };
    let d = Foo { n: 8, name: "d" };
    let e = Foo { n: 22, name: "e" };
    let f = Foo { n: 22, name: "f" };

    let data = [a, b, c, d, e, f];

    // `min` should return the left when the values are equal
    assert_eq!(data.iter().min(), Some(&a));
    assert_eq!(data.iter().min_by_key(|a| a.n), Some(&a));
    assert_eq!(cmp::min(a, b), a);
    assert_eq!(cmp::min(b, a), b);

    // `max` should return the right when the values are equal
    assert_eq!(data.iter().max(), Some(&f));
    assert_eq!(data.iter().max_by_key(|a| a.n), Some(&f));
    assert_eq!(cmp::max(e, f), f);
    assert_eq!(cmp::max(f, e), e);

    let mut presorted = data.to_vec();
    presorted.sort();
    assert_stable(&presorted);

    let mut presorted = data.to_vec();
    presorted.sort_by(|a, b| a.cmp(b));
    assert_stable(&presorted);

    // Assert that sorted and min/max are the same
    fn assert_stable<T: Ord + Debug>(presorted: &[T]) {
        for slice in presorted.windows(2) {
            let a = &slice[0];
            let b = &slice[1];

            assert_eq!(a, cmp::min(a, b));
            assert_eq!(b, cmp::max(a, b));
        }
    }
}
