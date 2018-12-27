// run-pass
#![feature(slice_patterns)]

fn main() {
    let x = [(), ()];

    // The subslice used to go out of bounds for zero-sized array items, check that this doesn't
    // happen anymore
    match x {
        [_, ref y..] => assert_eq!(&x[1] as *const (), &y[0] as *const ())
    }
}
