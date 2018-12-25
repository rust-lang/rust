// aux-build:double.rs
// aux-build:external-crate-var.rs

#![allow(unused)]

#[macro_use]
extern crate double;
#[macro_use]
extern crate external_crate_var;

struct Foo;

trait Trait {
    const CONST: u32;
    type Assoc;
}

impl Trait for Foo {
    const CONST: u32 = 0;
    type Assoc = Foo;
}

macro_rules! local { () => {
    // derive_Double outputs secondary copies of each definition
    // to test what the proc_macro sees.
    mod bar {
        #[derive(Double)]
        struct Bar($crate::Foo);
    }

    mod qself {
        #[derive(Double)]
        struct QSelf(<::Foo as $crate::Trait>::Assoc);
    }

    mod qself_recurse {
        #[derive(Double)]
        struct QSelfRecurse(<<$crate::Foo as $crate::Trait>::Assoc as $crate::Trait>::Assoc);
    }

    mod qself_in_const {
        #[derive(Double)]
        #[repr(u32)]
        enum QSelfInConst {
            Variant = <::Foo as $crate::Trait>::CONST,
        }
    }
} }

mod local {
    local!();
}

// and now repeat the above tests, using a macro defined in another crate

mod external {
    external!{}
}

fn main() {}
