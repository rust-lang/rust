// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: unresolved

mod circ1 {
    #[legacy_exports];
    use circ1::*;
    export f1;
    export f2;
    export common;
    fn f1() { debug!("f1"); }
    fn common() -> uint { return 0u; }
}

mod circ2 {
    #[legacy_exports];
    use circ2::*;
    export f1;
    export f2;
    export common;
    fn f2() { debug!("f2"); }
    fn common() -> uint { return 1u; }
}

mod test {
    #[legacy_exports];
    use circ1::*;

    fn test() { f1066(); }
}
