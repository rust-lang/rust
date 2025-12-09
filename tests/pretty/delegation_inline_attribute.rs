//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:delegation_inline_attribute.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

mod to_reuse {
    pub fn foo(x: usize) -> usize {
        x
    }
}

// Check that #[inline(hint)] is added to foo reuse
reuse to_reuse::foo as bar {
    self + 1
}

trait Trait {
    fn foo(&self) {}
    fn foo1(&self) {}
    fn foo2(&self) {}
    fn foo3(&self) {}
    fn foo4(&self) {}
}

impl Trait for u8 {}

struct S(u8);

mod to_import {
    pub fn check(arg: &u8) -> &u8 { arg }
}

impl Trait for S {
    // Check that #[inline(hint)] is added to foo reuse
    reuse Trait::foo {
        // Check that #[inline(hint)] is added to foo0 reuse inside another reuse
        reuse to_reuse::foo as foo0 {
            self + 1
        }

        // Check that #[inline(hint)] is added when other attributes present in inner reuse
        #[cold]
        #[must_use]
        #[deprecated]
        reuse to_reuse::foo as foo1 {
            self / 2
        }

        // Check that #[inline(never)] is preserved in inner reuse
        #[inline(never)]
        reuse to_reuse::foo as foo2 {
            self / 2
        }

        // Check that #[inline(always)] is preserved in inner reuse
        #[inline(always)]
        reuse to_reuse::foo as foo3 {
            self / 2
        }

        // Check that #[inline(never)] is preserved when there are other attributes in inner reuse
        #[cold]
        #[must_use]
        #[inline(never)]
        #[deprecated]
        reuse to_reuse::foo as foo4 {
            self / 2
        }
    }

    // Check that #[inline(hint)] is added when there are other attributes present in trait reuse
    #[cold]
    #[must_use]
    #[deprecated]
    reuse Trait::foo1 {
        self.0
    }

    // Check that #[inline(never)] is preserved in trait reuse
    #[inline(never)]
    reuse Trait::foo2 {
        self.0
    }

    // Check that #[inline(always)] is preserved in trait reuse
    #[inline(always)]
    reuse Trait::foo3 {
        self.0
    }

    // Check that #[inline(never)] is preserved when there are other attributes in trait reuse
    #[cold]
    #[must_use]
    #[inline(never)]
    #[deprecated]
    reuse Trait::foo4 {
        self.0
    }
}

fn main() {
}
