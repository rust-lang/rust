// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that moving a type definition within a source file does not affect
// re-compilation.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph -g

#![rustc_partition_reused(module="spans_in_type_debuginfo-structs", cfg="rpass2")]
#![rustc_partition_reused(module="spans_in_type_debuginfo-enums", cfg="rpass2")]

#![feature(rustc_attrs)]

mod structs {
    #[cfg(rpass1)]
    pub struct X {
        pub x: u32,
    }

    #[cfg(rpass2)]
    pub struct X {
        pub x: u32,
    }

    pub fn foo(x: X) -> u32 {
        x.x
    }
}

mod enums {
    #[cfg(rpass1)]
    pub enum X {
        A { x: u32 },
        B(u32),
    }

    #[cfg(rpass2)]
    pub enum X {
        A { x: u32 },
        B(u32),
    }

    pub fn foo(x: X) -> u32 {
        match x {
            X::A { x } => x,
            X::B(x) => x,
        }
    }
}

pub fn main() {
    let _ = structs::foo(structs::X { x: 1 });
    let _ = enums::foo(enums::X::A { x: 2 });
}
