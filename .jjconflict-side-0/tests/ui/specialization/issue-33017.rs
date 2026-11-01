// Test to ensure that trait bounds are properly
// checked on specializable associated types

#![allow(incomplete_features)]
#![feature(specialization)]

trait UncheckedCopy: Sized {
    type Output: From<Self> + Copy + Into<Self>;
}

impl<T> UncheckedCopy for T {
    default type Output = Self;
    //~^ ERROR: the trait bound `T: Copy` is not satisfied
}

fn unchecked_copy<T: UncheckedCopy>(other: &T::Output) -> T {
    (*other).into()
}

fn bug(origin: String) {
    // Turn the String into it's Output type...
    // Which we can just do by `.into()`, the assoc type states `From<Self>`.
    let origin_output = origin.into();

    // Make a copy of String::Output, which is a String...
    let mut copy: String = unchecked_copy::<String>(&origin_output);

    // Turn the Output type into a String again,
    // Which we can just do by `.into()`, the assoc type states `Into<Self>`.
    let mut origin: String = origin_output.into();

    // assert both Strings use the same buffer.
    assert_eq!(copy.as_ptr(), origin.as_ptr());

    // Any use of the copy we made becomes invalid,
    drop(origin);

    // OH NO! UB UB UB UB!
    copy.push_str(" world!");
    println!("{}", copy);
}

fn main() {
    bug(String::from("hello"));
}
