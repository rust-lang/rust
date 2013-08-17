// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// These are attributes of the implicit crate. Really this just needs to parse
// for completeness since .rs files linked from .rc files support this
// notation to specify their module's attributes
#[attr1 = "val"];
#[attr2 = "val"];
#[attr3];
#[attr4(attr5)];

// Special linkage attributes for the crate
#[link(name = "extra",
       vers = "0.1",
       uuid = "122bed0b-c19b-4b82-b0b7-7ae8aead7297",
       url = "http://rust-lang.org/src/extra")];

// These are attributes of the following mod
#[attr1 = "val"]
#[attr2 = "val"]
mod test_first_item_in_file_mod {}

mod test_single_attr_outer {
    #[attr = "val"]
    pub static x: int = 10;

    #[attr = "val"]
    pub fn f() { }

    #[attr = "val"]
    pub mod mod1 {}

    pub mod rustrt {
        #[attr = "val"]
        #[abi = "cdecl"]
        extern {}
    }
}

mod test_multi_attr_outer {
    #[attr1 = "val"]
    #[attr2 = "val"]
    pub static x: int = 10;

    #[attr1 = "val"]
    #[attr2 = "val"]
    pub fn f() { }

    #[attr1 = "val"]
    #[attr2 = "val"]
    pub mod mod1 {}

    pub mod rustrt {
        #[attr1 = "val"]
        #[attr2 = "val"]
        #[abi = "cdecl"]
        extern {}
    }

    #[attr1 = "val"]
    #[attr2 = "val"]
    struct t {x: int}
}

mod test_stmt_single_attr_outer {
    pub fn f() {
        #[attr = "val"]
        static x: int = 10;

        #[attr = "val"]
        fn f() { }

        #[attr = "val"]
        mod mod1 {
        }

        mod rustrt {
            #[attr = "val"]
            #[abi = "cdecl"]
            extern {
            }
        }
    }
}

mod test_stmt_multi_attr_outer {
    pub fn f() {

        #[attr1 = "val"]
        #[attr2 = "val"]
        static x: int = 10;

        #[attr1 = "val"]
        #[attr2 = "val"]
        fn f() { }

        /* FIXME: Issue #493
        #[attr1 = "val"]
        #[attr2 = "val"]
        mod mod1 {
        }

        pub mod rustrt {
            #[attr1 = "val"]
            #[attr2 = "val"]
            #[abi = "cdecl"]
            extern {
            }
        }
        */
    }
}

mod test_attr_inner {
    pub mod m {
        // This is an attribute of mod m
        #[attr = "val"];
    }
}

mod test_attr_inner_then_outer {
    pub mod m {
        // This is an attribute of mod m
        #[attr = "val"];
        // This is an attribute of fn f
        #[attr = "val"]
        fn f() { }
    }
}

mod test_attr_inner_then_outer_multi {
    pub mod m {
        // This is an attribute of mod m
        #[attr1 = "val"];
        #[attr2 = "val"];
        // This is an attribute of fn f
        #[attr1 = "val"]
        #[attr2 = "val"]
        fn f() { }
    }
}

mod test_distinguish_syntax_ext {
    extern mod extra;

    pub fn f() {
        fmt!("test%s", "s");
        #[attr = "val"]
        fn g() { }
    }
}

mod test_other_forms {
    #[attr]
    #[attr(word)]
    #[attr(attr(word))]
    #[attr(key1 = "val", key2 = "val", attr)]
    pub fn f() { }
}

mod test_foreign_items {
    pub mod rustrt {
        use std::libc;

        #[abi = "cdecl"]
        extern {
            #[attr];

            #[attr]
            fn rust_get_test_int() -> libc::intptr_t;
        }
    }
}

mod test_literals {
    #[str = "s"];
    #[char = 'c'];
    #[int = 100];
    #[uint = 100u];
    #[mach_int = 100u32];
    #[float = 1.0];
    #[mach_float = 1.0f32];
    #[nil = ()];
    #[bool = true];
    mod m {}
}

fn test_fn_inner() {
    #[inner_fn_attr];
}

pub fn main() { }
