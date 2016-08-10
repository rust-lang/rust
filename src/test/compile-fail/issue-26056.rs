// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MapLookup<Q> {
    type MapValue;
}

impl<K> MapLookup<K> for K {
    type MapValue = K;
}

trait Map: MapLookup<<Self as Map>::Key> {
    type Key;
}

impl<K> Map for K {
    type Key = K;
}


fn main() {
    let _ = &()
        as &Map<Key=u32,MapValue=u32>;
    //~^ ERROR E0038
    //~| NOTE the trait cannot use `Self` as a type parameter
    //~| NOTE the trait `Map` cannot be made into an object
}
