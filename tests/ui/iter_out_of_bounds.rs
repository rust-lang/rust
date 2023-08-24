#![deny(clippy::iter_out_of_bounds)]

fn opaque_empty_iter() -> impl Iterator<Item = ()> {
    std::iter::empty()
}

fn main() {
    for _ in [1, 2, 3].iter().skip(4) {
        //~^ ERROR: this `.skip()` call skips more items than the iterator will produce
        unreachable!();
    }
    for (i, _) in [1, 2, 3].iter().take(4).enumerate() {
        //~^ ERROR: this `.take()` call takes more items than the iterator will produce
        assert!(i <= 2);
    }

    #[allow(clippy::needless_borrow)]
    for _ in (&&&&&&[1, 2, 3]).iter().take(4) {}
    //~^ ERROR: this `.take()` call takes more items than the iterator will produce

    for _ in [1, 2, 3].iter().skip(4) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    // Should not lint
    for _ in opaque_empty_iter().skip(1) {}

    // Should not lint
    let empty: [i8; 0] = [];
    for _ in empty.iter().skip(1) {}

    let empty = std::iter::empty::<i8>;

    for _ in empty().skip(1) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    for _ in empty().take(1) {}
    //~^ ERROR: this `.take()` call takes more items than the iterator will produce

    for _ in std::iter::once(1).skip(2) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    for _ in std::iter::once(1).take(2) {}
    //~^ ERROR: this `.take()` call takes more items than the iterator will produce
}
