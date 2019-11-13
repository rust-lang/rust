// Regression test for issue #3645

fn main() {
    let n = 1;
    let a = [0; n];
    //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    let b = [0; ()];
    //~^ ERROR mismatched types
    //~| expected usize, found ()
    //~| expected type `usize`
    //~| found type `()`
    let c = [0; true];
    //~^ ERROR mismatched types
    //~| expected usize, found bool
    let d = [0; 0.5];
    //~^ ERROR mismatched types
    //~| expected usize, found floating-point number
    //~| expected type `usize`
    //~| found type `{float}`
    let e = [0; "foo"];
    //~^ ERROR mismatched types
    //~| expected usize, found reference
    //~| expected type `usize`
    //~| found reference `&'static str`
    let f = [0; -4_isize];
    //~^ ERROR mismatched types
    //~| expected usize, found isize
    let f = [0_usize; -1_isize];
    //~^ ERROR mismatched types
    //~| expected usize, found isize
    struct G {
        g: (),
    }
    let g = [0; G { g: () }];
    //~^ ERROR mismatched types
    //~| expected usize, found struct `main::G`
    //~| expected type `usize`
    //~| found struct `main::G`
}
