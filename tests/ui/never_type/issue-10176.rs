fn f() -> isize {
    (return 1, return 2)
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected type `isize`
//~| NOTE_NONVIRAL found tuple `(!, !)`
//~| NOTE_NONVIRAL expected `isize`, found `(!, !)`
}

fn main() {}
