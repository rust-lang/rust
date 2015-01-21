// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::IntoCow;

fn main() {
    <String as IntoCow>::into_cow("foo".to_string());
    //~^ ERROR wrong number of type arguments: expected 2, found 0
}

