// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the hash for `mod3::bar` changes when we change the
// `use` to something different.

// revisions: rpass1 rpass2 rpass3

#![feature(rustc_attrs)]

fn test<T>() { }

mod mod1 {
    pub struct Foo(pub u32);
}

mod mod2 {
    pub struct Foo(pub i64);
}

#[cfg(rpass1)]
mod mod3 {
    use test;
    use mod1::Foo;

    fn in_expr() {
        Foo(0);
    }

    fn in_type() {
        test::<Foo>();
    }
}

#[cfg(rpass2)]
mod mod3 {
    use mod1::Foo; // <-- Nothing changed, but reordered!
    use test;

    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    fn in_expr() {
        Foo(0);
    }

    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    fn in_type() {
        test::<Foo>();
    }
}

#[cfg(rpass3)]
mod mod3 {
    use test;
    use mod2::Foo; // <-- This changed!

    #[rustc_clean(label="Hir", cfg="rpass3")]
    #[rustc_dirty(label="HirBody", cfg="rpass3")]
    fn in_expr() {
        Foo(0);
    }

    #[rustc_clean(label="Hir", cfg="rpass3")]
    #[rustc_dirty(label="HirBody", cfg="rpass3")]
    fn in_type() {
        test::<Foo>();
    }
}

fn main() { }
