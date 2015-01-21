// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that associated types are `Sized`

trait Trait {
    type Output;

    fn is_sized(&self) -> Self::Output;
    fn wasnt_sized(&self) -> Self::Output { loop {} }
}

fn main() {}
