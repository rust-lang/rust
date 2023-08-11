//@aux-build:../../ui/auxiliary/proc_macros.rs
//@aux-build:helper.rs
//@revisions: allow_crates disallow_crates
//@[allow_crates] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/allow_crates
//@[disallow_crates] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/disallow_crates
#![allow(clippy::no_effect, unused)]
#![warn(clippy::absolute_paths)]
#![feature(decl_macro)]

extern crate helper;
#[macro_use]
extern crate proc_macros;

pub mod a {
    pub mod b {
        pub mod c {
            pub struct C;

            impl C {
                pub const ZERO: u32 = 0;
            }

            pub mod d {
                pub mod e {
                    pub mod f {
                        pub struct F;
                    }
                }
            }
        }

        pub struct B;
    }

    pub struct A;
}

fn main() {
    f32::max(1.0, 2.0);
    std::f32::MAX;
    core::f32::MAX;
    ::core::f32::MAX;
    crate::a::b::c::C;
    crate::a::b::c::d::e::f::F;
    crate::a::A;
    crate::a::b::B;
    crate::a::b::c::C::ZERO;
    helper::b::c::d::e::f();
    ::helper::b::c::d::e::f();
    fn b() -> a::b::B {
        todo!()
    }
    std::println!("a");
    let x = 1;
    std::ptr::addr_of!(x);
    // Test we handle max segments with `PathRoot` properly; this has 4 segments but we should say it
    // has 3
    ::std::f32::MAX;
    // Do not lint due to the above
    ::helper::a();
    // Do not lint
    helper::a();
    use crate::a::b::c::C;
    use a::b;
    use std::f32::MAX;
    a::b::c::d::e::f::F;
    b::c::C;
    fn a() -> a::A {
        todo!()
    }
    use a::b::c;

    fn c() -> c::C {
        todo!()
    }
    fn d() -> Result<(), ()> {
        todo!()
    }
    external! {
        crate::a::b::c::C::ZERO;
    }
    // For some reason, `path.span.from_expansion()` takes care of this for us
    with_span! {
        span
        crate::a::b::c::C::ZERO;
    }
    macro_rules! local_crate {
        () => {
            crate::a::b::c::C::ZERO;
        };
    }
    macro local_crate_2_0() {
        crate::a::b::c::C::ZERO;
    }
    local_crate!();
    local_crate_2_0!();
}
