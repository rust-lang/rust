// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:method `nap` is private

mod kitties {
    use std::uint;

    pub struct cat {
        priv meows : uint,

        how_hungry : int,
    }

    pub impl cat {
        priv fn nap(&self) { uint::range(1u, 10000u, |_i| false); }
    }

    pub fn cat(in_x : uint, in_y : int) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }
}

fn main() {
  let nyan : kitties::cat = kitties::cat(52u, 99);
  nyan.nap();
}
