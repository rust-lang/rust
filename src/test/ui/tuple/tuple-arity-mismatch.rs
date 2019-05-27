// Issue #6155

fn first((value, _): (isize, f64)) -> isize { value }

fn main() {
    let y = first ((1,2.0,3));
    //~^ ERROR mismatched types
    //~| expected type `(isize, f64)`
    //~| found type `(isize, f64, {integer})`
    //~| expected a tuple with 2 elements, found one with 3 elements

    let y = first ((1,));
    //~^ ERROR mismatched types
    //~| expected type `(isize, f64)`
    //~| found type `(isize,)`
    //~| expected a tuple with 2 elements, found one with 1 element
}
