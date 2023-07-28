#![allow(clippy::uninlined_format_args)]

use std::iter::repeat;
fn square_is_lower_64(x: &u32) -> bool {
    x * x < 64
}

#[allow(clippy::maybe_infinite_iter)]
#[deny(clippy::infinite_iter)]
fn infinite_iters() {
    repeat(0_u8).collect::<Vec<_>>();
    // infinite iter
    (0..8_u32).take_while(square_is_lower_64).cycle().count();
    // infinite iter
    (0..8_u64).chain(0..).max();
    // infinite iter
    (0_usize..)
        .chain([0usize, 1, 2].iter().cloned())
        .skip_while(|x| *x != 42)
        .min();
    // infinite iter
    (0..8_u32)
        .rev()
        .cycle()
        .map(|x| x + 1_u32)
        .for_each(|x| println!("{}", x));
    // infinite iter
    (0..3_u32).flat_map(|x| x..).sum::<u32>();
    // infinite iter
    (0_usize..).flat_map(|x| 0..x).product::<usize>();
    // infinite iter
    (0_u64..).filter(|x| x % 2 == 0).last();
    // not an infinite, because ranges are double-ended
    (0..42_u64).by_ref().last();
    // iterator is not exhausted
    (0..).next();
}

#[deny(clippy::maybe_infinite_iter)]
fn potential_infinite_iters() {
    // maybe infinite iter
    (0..).zip((0..).take_while(square_is_lower_64)).count();
    // maybe infinite iter
    repeat(42).take_while(|x| *x == 42).chain(0..42).max();
    // maybe infinite iter
    (1..)
        .scan(0, |state, x| {
            *state += x;
            Some(*state)
        })
        .min();
    // maybe infinite iter
    (0..).find(|x| *x == 24);
    // maybe infinite iter
    (0..).position(|x| x == 24);
    // maybe infinite iter
    (0..).any(|x| x == 24);
    // maybe infinite iter
    (0..).all(|x| x == 24);

    // not infinite
    (0..).zip(0..42).take_while(|&(x, _)| x != 42).count();
    // iterator is not exhausted
    repeat(42).take_while(|x| *x == 42).next();
}

fn main() {
    infinite_iters();
    potential_infinite_iters();
}

mod finite_collect {
    use std::collections::HashSet;

    struct C;
    impl FromIterator<i32> for C {
        fn from_iter<I: IntoIterator<Item = i32>>(iter: I) -> Self {
            C
        }
    }

    fn check_collect() {
        // Infinite iter
        let _: HashSet<i32> = (0..).collect();

        // Some data structures don't collect infinitely, such as `ArrayVec`
        let _: C = (0..).collect();
    }
}
