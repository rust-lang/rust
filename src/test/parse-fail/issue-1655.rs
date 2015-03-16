// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:expected `[`, found `vec`
mod blade_runner {
    #vec[doc(
        brief = "Blade Runner is probably the best movie ever",
        desc = "I like that in the world of Blade Runner it is always
                raining, and that it's always night time. And Aliens
                was also a really good movie.

                Alien 3 was crap though."
    )]
}
