// Regression test for issue #4968

const A: (isize,isize) = (4,2);
fn main() {
    match 42 { A => () }
    //~^ ERROR mismatched types
    //~| expected type `{integer}`
    //~| found tuple `(isize, isize)`
    //~| expected integer, found `(isize, isize)`
}
