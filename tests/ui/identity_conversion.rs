// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::identity_conversion)]

fn test_generic<T: Copy>(val: T) -> T {
    let _ = T::from(val);
    val.into()
}

fn test_generic2<T: Copy + Into<i32> + Into<U>, U: From<T>>(val: T) {
    // ok
    let _: i32 = val.into();
    let _: U = val.into();
    let _ = U::from(val);
}

fn test_questionmark() -> Result<(), ()> {
    {
        let _: i32 = 0i32.into();
        Ok(Ok(()))
    }??;
    Ok(())
}

fn main() {
    test_generic(10i32);
    test_generic2::<i32, i32>(10i32);
    test_questionmark().unwrap();

    let _: String = "foo".into();
    let _: String = From::from("foo");
    let _ = String::from("foo");
    #[allow(clippy::identity_conversion)]
    {
        let _: String = "foo".into();
        let _ = String::from("foo");
        let _ = "".lines().into_iter();
    }

    let _: String = "foo".to_string().into();
    let _: String = From::from("foo".to_string());
    let _ = String::from("foo".to_string());
    let _ = String::from(format!("A: {:04}", 123));
    let _ = "".lines().into_iter();
    let _ = vec![1, 2, 3].into_iter().into_iter();
    let _: String = format!("Hello {}", "world").into();
}
