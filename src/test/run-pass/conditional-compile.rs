// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Crate use statements
#[cfg(bogus)]
use flippity;

#[cfg(bogus)]
static b: bool = false;

static b: bool = true;

mod rustrt {
    #[cfg(bogus)]
    #[abi = "cdecl"]
    pub extern {
        // This symbol doesn't exist and would be a link error if this
        // module was translated
        pub fn bogus();
    }
    
    #[abi = "cdecl"]
    pub extern {}
}

#[cfg(bogus)]
type t = int;

type t = bool;

#[cfg(bogus)]
enum tg { foo, }

enum tg { bar, }

#[cfg(bogus)]
struct r {
  i: int,
}

#[cfg(bogus)]
fn r(i:int) -> r {
    r {
        i: i
    }
}

struct r {
  i: int,
}

fn r(i:int) -> r {
    r {
        i: i
    }
}

#[cfg(bogus)]
mod m {
    // This needs to parse but would fail in typeck. Since it's not in
    // the current config it should not be typechecked.
    pub fn bogus() { return 0; }
}

mod m {
    // Submodules have slightly different code paths than the top-level
    // module, so let's make sure this jazz works here as well
    #[cfg(bogus)]
    pub fn f() { }

    pub fn f() { }
}

// Since the bogus configuration isn't defined main will just be
// parsed, but nothing further will be done with it
#[cfg(bogus)]
pub fn main() { fail!() }

pub fn main() {
    // Exercise some of the configured items in ways that wouldn't be possible
    // if they had the bogus definition
    assert!((b));
    let x: t = true;
    let y: tg = bar;

    test_in_fn_ctxt();
}

fn test_in_fn_ctxt() {
    #[cfg(bogus)]
    fn f() { fail!() }
    fn f() { }
    f();

    #[cfg(bogus)]
    static i: int = 0;
    static i: int = 1;
    assert!((i == 1));
}

mod test_foreign_items {
    pub mod rustrt {
        #[abi = "cdecl"]
        pub extern {
            #[cfg(bogus)]
            pub fn rust_get_stdin() -> ~str;
            pub fn rust_get_stdin() -> ~str;
        }
    }
}

mod test_use_statements {
    #[cfg(bogus)]
    use flippity_foo;
}

mod test_methods {
    struct Foo {
        bar: uint
    }

    impl Fooable for Foo {
        #[cfg(bogus)]
        fn what(&self) { }

        fn what(&self) { }

        #[cfg(bogus)]
        fn the(&self) { }

        fn the(&self) { }
    }

    trait Fooable {
        #[cfg(bogus)]
        fn what(&self);

        fn what(&self);

        #[cfg(bogus)]
        fn the(&self);

        fn the(&self);
    }
}
