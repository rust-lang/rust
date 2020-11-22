#![feature(precise_pointer_size_matching)]
#![feature(exclusive_range_pattern)]

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

fn main() {
    match 0usize {
        0 ..= usize::MAX => {}
    }

    match 0isize {
        isize::MIN ..= isize::MAX => {}
    }

    m!(0usize, 0..=usize::MAX);
    m!(0usize, 0..5 | 5..=usize::MAX);
    m!(0usize, 0..usize::MAX | usize::MAX);
    m!((0usize, true), (0..5, true) | (5..=usize::MAX, true) | (0..=usize::MAX, false));

    m!(0isize, isize::MIN..=isize::MAX);
    m!(0isize, isize::MIN..5 | 5..=isize::MAX);
    m!(0isize, isize::MIN..isize::MAX | isize::MAX);
    m!((0isize, true), (isize::MIN..5, true)
        | (5..=isize::MAX, true) | (isize::MIN..=isize::MAX, false));

    match 0isize {
        isize::MIN ..= -1 => {}
        0 => {}
        1 ..= isize::MAX => {}
    }

    match 7usize {}
    //~^ ERROR non-exhaustive patterns
}
