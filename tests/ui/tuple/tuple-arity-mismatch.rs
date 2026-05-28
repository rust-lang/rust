// Issue #6155

//@ dont-require-annotations: NOTE

fn first((value, _): (isize, f64)) -> isize { value }

fn main() {
    let y = first ((1,2.0,3));
    //~^ ERROR mismatched types
    //~| NOTE expected tuple `(isize, f64)`
    //~| NOTE found tuple `(isize, f64, {integer})`
    //~| NOTE expected a tuple with 2 elements, found one with 3 elements

    let y = first ((1,));
    //~^ ERROR mismatched types
    //~| NOTE expected tuple `(isize, f64)`
    //~| NOTE found tuple `(isize,)`
    //~| NOTE expected a tuple with 2 elements, found one with 1 element
}
