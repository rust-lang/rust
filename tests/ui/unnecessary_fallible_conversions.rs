#![warn(clippy::unnecessary_fallible_conversions)]

fn main() {
    // --- TryFromMethod `T::try_from(u)` ---

    let _: i64 = 0i32.try_into().unwrap();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _: i64 = 0i32.try_into().expect("can't happen");
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    // --- TryFromFunction `T::try_from(U)` ---

    let _ = i64::try_from(0i32).unwrap();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _ = i64::try_from(0i32).expect("can't happen");
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    // --- TryIntoFunction `U::try_into(t)` ---

    let _: i64 = i32::try_into(0).unwrap();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _: i64 = i32::try_into(0i32).expect("can't happen");
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    // --- TryFromFunction `<T as TryFrom<U>>::try_from(U)` ---

    let _ = <i64 as TryFrom<i32>>::try_from(0).unwrap();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _ = <i64 as TryFrom<i32>>::try_from(0).expect("can't happen");
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    // --- TryIntoFunction `<U as TryInto<_>>::try_into(U)` ---

    let _: i64 = <i32 as TryInto<_>>::try_into(0).unwrap();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _: i64 = <i32 as TryInto<_>>::try_into(0).expect("can't happen");
    //~^ ERROR: use of a fallible conversion when an infallible one could be used
}
