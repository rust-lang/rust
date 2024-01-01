#![warn(clippy::unnecessary_fallible_conversions)]

fn main() {
    let _: i64 = 0i32.try_into().unwrap();
    let _: i64 = 0i32.try_into().expect("can't happen");

    let _ = i64::try_from(0i32).unwrap();
    let _ = i64::try_from(0i32).expect("can't happen");

    let _: i64 = i32::try_into(0).unwrap();
    let _: i64 = i32::try_into(0i32).expect("can't happen");

    let _ = <i64 as TryFrom<i32>>::try_from(0).unwrap();
    let _ = <i64 as TryFrom<i32>>::try_from(0).expect("can't happen");

    let _: i64 = <i32 as TryInto<_>>::try_into(0).unwrap();
    let _: i64 = <i32 as TryInto<_>>::try_into(0).expect("can't happen");
}
