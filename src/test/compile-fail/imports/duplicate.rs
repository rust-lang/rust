// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub fn foo() {}
}

mod b {
    pub fn foo() {}
}

mod c {
    pub use a::foo;
}

mod d {
    use a::foo; //~ NOTE previous import
    use a::foo; //~ ERROR `foo` has already been imported
                //~| NOTE already imported
}

mod e {
    pub use a::*;
    pub use c::*; // ok
}

mod f {
    pub use a::*; //~ NOTE `foo` could refer to the name imported here
    pub use b::*; //~ NOTE `foo` could also refer to the name imported here
}

mod g {
    pub use a::*; //~ NOTE `foo` could refer to the name imported here
    pub use f::*; //~ NOTE `foo` could also refer to the name imported here
}

fn main() {
    e::foo();
    f::foo(); //~ ERROR `foo` is ambiguous
              //~| NOTE consider adding an explicit import of `foo` to disambiguate
    g::foo(); //~ ERROR `foo` is ambiguous
              //~| NOTE consider adding an explicit import of `foo` to disambiguate
}

mod ambiguous_module_errors {
    pub mod m1 { pub use super::m1 as foo; }
    pub mod m2 { pub use super::m2 as foo; }

    use self::m1::*; //~ NOTE
                     //~| NOTE
    use self::m2::*; //~ NOTE
                     //~| NOTE

    use self::foo::bar; //~ ERROR `foo` is ambiguous
                        //~| NOTE

    fn f() {
        foo::bar(); //~ ERROR `foo` is ambiguous
                    //~| NOTE
    }
}
