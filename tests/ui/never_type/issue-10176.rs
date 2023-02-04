fn f() -> isize {
    (return 1, return 2)
//~^ ERROR mismatched types
//~| expected type `isize`
//~| found tuple `(!, !)`
//~| expected `isize`, found `(!, !)`
}

fn main() {}
