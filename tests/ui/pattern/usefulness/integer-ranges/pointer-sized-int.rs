//@ revisions: deny
#![allow(overlapping_range_endpoints)]

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

#[rustfmt::skip]
fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns
        0..=usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns
        isize::MIN..=isize::MAX => {}
    }

    m!(0usize, 0..=usize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(0usize, 0..5 | 5..=usize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(0usize, 0..usize::MAX | usize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!((0usize, true), (0..5, true) | (5..=usize::MAX, true) | (0..=usize::MAX, false));
    //~^ ERROR non-exhaustive patterns

    m!(0usize, 0..);
    m!(0usize, 0..5 | 5..);
    m!(0usize, ..5 | 5..);
    m!((0usize, true), (0..5, true) | (5.., true) | (0.., false));
    m!(0usize, 0..=usize::MAX | usize::MAX..);

    m!(0isize, isize::MIN..=isize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(0isize, isize::MIN..5 | 5..=isize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(0isize, isize::MIN..=-1 | 0 | 1..=isize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(0isize, isize::MIN..isize::MAX | isize::MAX);
    //~^ ERROR non-exhaustive patterns
    m!(
        (0isize, true),
        (isize::MIN..5, true) | (5..=isize::MAX, true) | (isize::MIN..=isize::MAX, false)
    );
    //~^^^ ERROR non-exhaustive patterns

    m!(0isize, ..0 | 0..);
    m!(0isize, ..5 | 5..);
    m!((0isize, true), (..5, true)
        | (5.., true) | (..0 | 0.., false));
    m!(0isize, ..=isize::MIN | isize::MIN..=isize::MAX | isize::MAX..);

    match 7usize {}
    //~^ ERROR non-exhaustive patterns
}
