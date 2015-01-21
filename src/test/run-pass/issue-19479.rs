// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Base {}
trait AssocA {
    type X: Base;
}
trait AssocB {
    type Y: Base;
}
impl<T: AssocA> AssocB for T {
    type Y = <T as AssocA>::X;
}

fn main() {}
