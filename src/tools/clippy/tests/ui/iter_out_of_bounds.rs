//@no-rustfix

#![deny(clippy::iter_out_of_bounds)]
#![allow(clippy::useless_vec)]

fn opaque_empty_iter() -> impl Iterator<Item = ()> {
    std::iter::empty()
}

fn main() {
    #[allow(clippy::never_loop)]
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

    for _ in [1; 3].iter().skip(4) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    // Should not lint
    for _ in opaque_empty_iter().skip(1) {}

    for _ in vec![1, 2, 3].iter().skip(4) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    for _ in vec![1; 3].iter().skip(4) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    let x = [1, 2, 3];
    for _ in x.iter().skip(4) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    let n = 4;
    for _ in x.iter().skip(n) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    let empty = std::iter::empty::<i8>;

    for _ in empty().skip(1) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    for _ in empty().take(1) {}
    //~^ ERROR: this `.take()` call takes more items than the iterator will produce

    for _ in std::iter::once(1).skip(2) {}
    //~^ ERROR: this `.skip()` call skips more items than the iterator will produce

    for _ in std::iter::once(1).take(2) {}
    //~^ ERROR: this `.take()` call takes more items than the iterator will produce

    for x in [].iter().take(1) {
        //~^ ERROR: this `.take()` call takes more items than the iterator will produce
        let _: &i32 = x;
    }

    // ok, not out of bounds
    for _ in [1].iter().take(1) {}
    for _ in [1, 2, 3].iter().take(2) {}
    for _ in [1, 2, 3].iter().skip(2) {}
}
