// revisions: allow deny
#![feature(exclusive_range_pattern)]
#![cfg_attr(allow, feature(precise_pointer_size_matching))]

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

fn main() {
    match 0usize {
        //[deny]~^ ERROR non-exhaustive patterns
        0 ..= usize::MAX => {}
    }

    match 0isize {
        //[deny]~^ ERROR non-exhaustive patterns
        isize::MIN ..= isize::MAX => {}
    }

    m!(0usize, 0..=usize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!(0usize, 0..5 | 5..=usize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!(0usize, 0..usize::MAX | usize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!((0usize, true), (0..5, true) | (5..=usize::MAX, true) | (0..=usize::MAX, false));
    //[deny]~^ ERROR non-exhaustive patterns

    m!(0isize, isize::MIN..=isize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!(0isize, isize::MIN..5 | 5..=isize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!(0isize, isize::MIN..isize::MAX | isize::MAX);
    //[deny]~^ ERROR non-exhaustive patterns
    m!((0isize, true), (isize::MIN..5, true)
        | (5..=isize::MAX, true) | (isize::MIN..=isize::MAX, false));
    //[deny]~^^ ERROR non-exhaustive patterns

    match 0isize {
        //[deny]~^ ERROR non-exhaustive patterns
        isize::MIN ..= -1 => {}
        0 => {}
        1 ..= isize::MAX => {}
    }

    match 7usize {}
    //~^ ERROR non-exhaustive patterns
}
