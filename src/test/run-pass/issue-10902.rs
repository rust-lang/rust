// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod two_tuple {
    pub trait T {}
    pub struct P<'a>(&'a (T + 'a), &'a (T + 'a));
    pub fn f<'a>(car: &'a T, cdr: &'a T) -> P<'a> {
        P(car, cdr)
    }
}

pub mod two_fields {
    pub trait T {}
    pub struct P<'a> { car: &'a (T + 'a), cdr: &'a (T + 'a) }
    pub fn f<'a>(car: &'a T, cdr: &'a T) -> P<'a> {
        P{ car: car, cdr: cdr }
    }
}

fn main() {}
