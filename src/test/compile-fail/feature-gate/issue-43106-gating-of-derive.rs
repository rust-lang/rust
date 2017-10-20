// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `#![derive]` raises errors when it occurs at contexts other than ADT
// definitions.

#![derive(Debug)]
//~^ ERROR `derive` may only be applied to structs, enums and unions

#[derive(Debug)]
//~^ ERROR `derive` may only be applied to structs, enums and unions
mod derive {
    mod inner { #![derive(Debug)] }
    //~^ ERROR `derive` may only be applied to structs, enums and unions

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    fn derive() { }

    #[derive(Copy, Clone)] // (can't derive Debug for unions)
    union U { f: i32 }

    #[derive(Debug)]
    struct S;

    #[derive(Debug)]
    enum E { }

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    type T = S;

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    impl S { }
}
