// Attempt to coerce from unsized to sized.

#![feature(unsized_tuple_coercion)]

struct Fat<T: ?Sized> {
    ptr: T
}

pub fn main() {
    // With a vec of isizes.
    let f1: &Fat<[isize]> = &Fat { ptr: [1, 2, 3] };
    let f2: &Fat<[isize; 3]> = f1;
    //~^ ERROR mismatched types
    //~| expected array `[isize; 3]`, found slice `[isize]`
    //~| expected reference `&Fat<[isize; 3]>`
    //~| found reference `&Fat<[isize]>`

    // Tuple with a vec of isizes.
    let f1: &([isize],) = &([1, 2, 3],);
    let f2: &([isize; 3],) = f1;
    //~^ ERROR mismatched types
    //~| expected array `[isize; 3]`, found slice `[isize]`
    //~| expected reference `&([isize; 3],)`
    //~| found reference `&([isize],)`
}
