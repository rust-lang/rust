// run-pass

use std::cmp::Ordering;

// the following methods of core::cmp::Ordering are const:
//  - reverse
//  - then

fn main() {
    const REVERSE : Ordering = Ordering::Greater.reverse();
    assert_eq!(REVERSE, Ordering::Less);

    const THEN : Ordering = Ordering::Equal.then(REVERSE);
    assert_eq!(THEN, Ordering::Less);
}
