#![allow(clippy::iter_next_slice, clippy::needless_return)]

fn no_break_or_continue_loop() {
    for i in [1, 2, 3].iter() {
        //~^ never_loop
        return;
    }
}

fn no_break_or_continue_loop_outer() {
    for i in [1, 2, 3].iter() {
        //~^ never_loop
        return;
        loop {
            if true {
                continue;
            }
        }
    }
}
