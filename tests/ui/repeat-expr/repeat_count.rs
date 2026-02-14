// Regression test for issue #3645

//@ dont-require-annotations: NOTE

fn main() {
    let n = 1;
    let a = [0; n];
    //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    let b = [0; ()];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `()`
    let c = [0; true];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `bool`
    let d = [0; 0.5];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found floating-point number
    let e = [0; "foo"];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `&str`
    let f = [0; -4_isize];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `isize`
    //~| NOTE `-4_isize` cannot fit into type `usize`
    let g = [0_usize; -1_isize];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `isize`
    //~| NOTE `-1_isize` cannot fit into type `usize`
    let h = [0; 4u8];
    //~^ ERROR the constant `4` is not of type `usize`
    //~| NOTE expected `usize`, found `u8`
    //~| NOTE the length of array `[{integer}; 4]` must be type `usize`
    //~| ERROR mismatched types
    //~| NOTE expected `usize`, found `u8`
    struct I {
        i: (),
    }
    let i = [0; I { i: () }];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `I`
}
