// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::*; //~ ERROR: unresolved import `self::*` [E0432]
             //~^ Cannot glob-import a module into itself.

mod foo {
    use foo::*; //~ ERROR: unresolved import `foo::*` [E0432]
                //~^ Cannot glob-import a module into itself.

    mod bar {
        use super::bar::*;
        //~^ ERROR: unresolved import `super::bar::*` [E0432]
        //~| Cannot glob-import a module into itself.
    }

}

fn main() {
}
