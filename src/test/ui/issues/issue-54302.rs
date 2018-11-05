// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Deserialize<'de> {}

trait DeserializeOwned: for<'de> Deserialize<'de> {}
impl<T> DeserializeOwned for T where T: for<'de> Deserialize<'de> {}

// Based on this impl, `&'static str` only implements Deserialize<'static>.
// It does not implement for<'de> Deserialize<'de>.
impl<'de: 'a, 'a> Deserialize<'de> for &'a str {}

fn main() {
    // Then why does it implement DeserializeOwned? This compiles.
    fn assert_deserialize_owned<T: DeserializeOwned>() {}
    assert_deserialize_owned::<&'static str>();
    //~^ ERROR the requirement `for<'de> 'de : ` is not satisfied

    // It correctly does not implement for<'de> Deserialize<'de>.
    //fn assert_hrtb<T: for<'de> Deserialize<'de>>() {}
    //assert_hrtb::<&'static str>();
}
