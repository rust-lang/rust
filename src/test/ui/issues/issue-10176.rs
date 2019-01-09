fn f() -> isize {
    (return 1, return 2)
//~^ ERROR mismatched types
//~| expected type `isize`
//~| found type `(!, !)`
//~| expected isize, found tuple
}

fn main() {}
