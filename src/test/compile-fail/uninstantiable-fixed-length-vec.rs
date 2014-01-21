// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue #11659, the compiler needs to know that a fixed length vector
// always requires instantiable contents to instantiable itself
// (unlike a ~[] vector which can have length zero).

// ~ to avoid infinite size.
struct Uninstantiable { //~ ERROR cannot be instantiated without an instance of itself
    p: ~[Uninstantiable, .. 1]
}

struct Instantiable { p: ~[Instantiable, .. 0] }


fn main() {
    let _ = None::<Uninstantiable>;
    let _ = Instantiable { p: ~([]) };
}
