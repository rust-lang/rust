// Attempt to coerce from unsized to sized.

struct Fat<T: ?Sized> {
    ptr: T
}

pub fn main() {
    // With a vec of isizes.
    let f1: &Fat<[isize]> = &Fat { ptr: [1, 2, 3] };
    let f2: &Fat<[isize; 3]> = f1;
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `&Fat<[isize; 3]>`, found `&Fat<[isize]>`
    //~| NOTE_NONVIRAL expected reference `&Fat<[isize; 3]>`
    //~| NOTE_NONVIRAL found reference `&Fat<[isize]>`
}
