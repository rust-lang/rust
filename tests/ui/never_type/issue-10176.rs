fn f() -> isize { //~ NOTE expected `isize` because of return type
    (return 1, return 2)
//~^ ERROR mismatched types
//~| NOTE expected type `isize`
//~| NOTE found tuple `(_, _)`
//~| NOTE expected `isize`, found `(_, _)`
}

fn main() {}
