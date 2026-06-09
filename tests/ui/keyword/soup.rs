//@ edition:2024
//@ check-pass

#![allow(unused_imports)]
#![allow(missing_abi)]
#![allow(unused_macros)]
#![allow(non_camel_case_types)]
#![allow(unreachable_code)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_must_use)]

// all 48 keywords in 300 characters
mod x {
    pub(super) struct X;
    use Ok;
    impl X {
        pub(in crate) async fn x(self: Self, x: &'static &'_ dyn for<> Fn()) where {
            unsafe extern { safe fn x(); }
            macro_rules! x { () => {}; }
            if 'x: loop {
                return match while let true = break 'x false { continue } {
                    ref x => { &raw mut x; async { const { enum A {} } }.await as () },
                };
            } { type x = X; } else { move || { trait x { } union B { x: () } }; }
        }
    }
}

fn main() {}
