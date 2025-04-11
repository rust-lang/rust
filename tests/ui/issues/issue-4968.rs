// Regression test for issue #4968

const A: (isize,isize) = (4,2);
fn main() {
    match 42 { A => () }
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected type `{integer}`
    //~| NOTE_NONVIRAL found tuple `(isize, isize)`
    //~| NOTE_NONVIRAL expected integer, found `(isize, isize)`
}
