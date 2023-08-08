#![deny(clippy::useless_conversion)]
#![allow(clippy::needless_if)]

fn test_generic<T: Copy>(val: T) -> T {
    let _ = T::try_from(val).unwrap();
    val.try_into().unwrap()
}

fn test_generic2<T: Copy + Into<i32> + Into<U>, U: From<T>>(val: T) {
    // ok
    let _: i32 = val.try_into().unwrap();
    let _: U = val.try_into().unwrap();
    let _ = U::try_from(val).unwrap();
}

fn main() {
    test_generic(10i32);
    test_generic2::<i32, i32>(10i32);

    let _: String = "foo".try_into().unwrap();
    let _: String = TryFrom::try_from("foo").unwrap();
    let _ = String::try_from("foo").unwrap();
    #[allow(clippy::useless_conversion)]
    {
        let _ = String::try_from("foo").unwrap();
        let _: String = "foo".try_into().unwrap();
    }
    let _: String = "foo".to_string().try_into().unwrap();
    let _: String = TryFrom::try_from("foo".to_string()).unwrap();
    let _ = String::try_from("foo".to_string()).unwrap();
    let _ = String::try_from(format!("A: {:04}", 123)).unwrap();
    let _: String = format!("Hello {}", "world").try_into().unwrap();
    let _: String = String::new().try_into().unwrap();
    let _: String = match String::from("_").try_into() {
        Ok(a) => a,
        Err(_) => String::new(),
    };
    // FIXME this is a false negative
    #[allow(clippy::cmp_owned)]
    if String::from("a") == TryInto::<String>::try_into(String::from("a")).unwrap() {}
}
