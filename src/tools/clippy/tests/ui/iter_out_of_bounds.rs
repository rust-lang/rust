#![deny(clippy::iter_out_of_bounds)]
#![allow(clippy::useless_vec)]

fn opaque_empty_iter() -> impl Iterator<Item = ()> {
    std::iter::empty()
}

fn main() {
    #[allow(clippy::never_loop)]
    for _ in [1, 2, 3].iter().skip(4) {
        //~^ iter_out_of_bounds

        unreachable!();
    }
    for (i, _) in [1, 2, 3].iter().take(4).enumerate() {
        //~^ iter_out_of_bounds

        assert!(i <= 2);
    }

    #[allow(clippy::needless_borrow)]
    for _ in (&&&&&&[1, 2, 3]).iter().take(4) {}
    //~^ iter_out_of_bounds

    for _ in [1, 2, 3].iter().skip(4) {}
    //~^ iter_out_of_bounds

    for _ in [1; 3].iter().skip(4) {}
    //~^ iter_out_of_bounds

    // Should not lint
    for _ in opaque_empty_iter().skip(1) {}

    for _ in vec![1, 2, 3].iter().skip(4) {}
    //~^ iter_out_of_bounds

    for _ in vec![1; 3].iter().skip(4) {}
    //~^ iter_out_of_bounds

    let x = [1, 2, 3];
    for _ in x.iter().skip(4) {}
    //~^ iter_out_of_bounds

    let n = 4;
    for _ in x.iter().skip(n) {}
    //~^ iter_out_of_bounds

    let empty = std::iter::empty::<i8>;

    for _ in empty().skip(1) {}
    //~^ iter_out_of_bounds

    for _ in empty().take(1) {}
    //~^ iter_out_of_bounds

    for _ in std::iter::once(1).skip(2) {}
    //~^ iter_out_of_bounds

    for _ in std::iter::once(1).take(2) {}
    //~^ iter_out_of_bounds

    for x in [].iter().take(1) {
        //~^ iter_out_of_bounds

        let _: &i32 = x;
    }

    // ok, not out of bounds
    for _ in [1].iter().take(1) {}
    for _ in [1, 2, 3].iter().take(2) {}
    for _ in [1, 2, 3].iter().skip(2) {}
}
