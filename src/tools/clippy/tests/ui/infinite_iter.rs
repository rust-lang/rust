#![allow(clippy::uninlined_format_args)]

use std::iter::repeat;
fn square_is_lower_64(x: &u32) -> bool {
    x * x < 64
}

#[allow(clippy::maybe_infinite_iter)]
#[deny(clippy::infinite_iter)]
fn infinite_iters() {
    repeat(0_u8).collect::<Vec<_>>();
    //~^ ERROR: infinite iteration detected
    // infinite iter
    (0..8_u32).take_while(square_is_lower_64).cycle().count();
    //~^ ERROR: infinite iteration detected
    // infinite iter
    (0..8_u64).chain(0..).max();
    //~^ ERROR: infinite iteration detected
    // infinite iter
    (0_usize..)
        .chain([0usize, 1, 2].iter().cloned())
        .skip_while(|x| *x != 42)
        .min();
    // infinite iter
    (0..8_u32)
        //~^ ERROR: infinite iteration detected
        .rev()
        .cycle()
        .map(|x| x + 1_u32)
        .for_each(|x| println!("{}", x));
    // infinite iter
    (0..3_u32).flat_map(|x| x..).sum::<u32>();
    // infinite iter
    (0_usize..).flat_map(|x| 0..x).product::<usize>();
    //~^ ERROR: infinite iteration detected
    // infinite iter
    (0_u64..).filter(|x| x % 2 == 0).last();
    //~^ ERROR: infinite iteration detected
    // not an infinite, because ranges are double-ended
    (0..42_u64).by_ref().last();
    // iterator is not exhausted
    (0..).next();
}

#[deny(clippy::maybe_infinite_iter)]
fn potential_infinite_iters() {
    // maybe infinite iter
    (0..).zip((0..).take_while(square_is_lower_64)).count();
    //~^ ERROR: possible infinite iteration detected
    // maybe infinite iter
    repeat(42).take_while(|x| *x == 42).chain(0..42).max();
    //~^ ERROR: possible infinite iteration detected
    // maybe infinite iter
    (1..)
        //~^ ERROR: possible infinite iteration detected
        .scan(0, |state, x| {
            *state += x;
            Some(*state)
        })
        .min();
    // maybe infinite iter
    (0..).find(|x| *x == 24);
    //~^ ERROR: possible infinite iteration detected
    // maybe infinite iter
    (0..).position(|x| x == 24);
    //~^ ERROR: possible infinite iteration detected
    // maybe infinite iter
    (0..).any(|x| x == 24);
    //~^ ERROR: possible infinite iteration detected
    // maybe infinite iter
    (0..).all(|x| x == 24);
    //~^ ERROR: possible infinite iteration detected

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
        //~^ ERROR: infinite iteration detected
        //~| NOTE: `#[deny(clippy::infinite_iter)]` on by default

        // Some data structures don't collect infinitely, such as `ArrayVec`
        let _: C = (0..).collect();
    }
}
