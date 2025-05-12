fn f() -> isize { //~ NOTE expected `isize` because of return type
    (return 1, return 2)
//~^ ERROR mismatched types
//~| NOTE expected type `isize`
//~| NOTE found tuple `(!, !)`
//~| NOTE expected `isize`, found `(!, !)`
}

fn main() {}
