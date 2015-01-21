// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum test {
    div_zero = 1/0, //~ERROR expected constant: attempted to divide by zero
    rem_zero = 1%0  //~ERROR expected constant: attempted remainder with a divisor of zero
}

fn main() {}
