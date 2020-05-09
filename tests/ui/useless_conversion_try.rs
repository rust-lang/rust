#![deny(clippy::useless_conversion)]

use std::convert::TryFrom;

fn test_generic<T: Copy>(val: T) -> T {
    T::try_from(val).unwrap()
}

fn test_generic2<T: Copy + Into<i32> + Into<U>, U: From<T>>(val: T) {
    let _ = U::try_from(val).unwrap();
}

fn main() {
    test_generic(10i32);
    test_generic2::<i32, i32>(10i32);

    let _: String = TryFrom::try_from("foo").unwrap();
    let _ = String::try_from("foo").unwrap();
    #[allow(clippy::useless_conversion)]
    let _ = String::try_from("foo").unwrap();

    let _: String = TryFrom::try_from("foo".to_string()).unwrap();
    let _ = String::try_from("foo".to_string()).unwrap();
    let _ = String::try_from(format!("A: {:04}", 123)).unwrap();
}
