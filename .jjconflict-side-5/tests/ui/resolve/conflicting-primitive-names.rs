//@ check-pass
#![allow(non_camel_case_types)]
#![allow(unused)]

// Ensure that primitives do not interfere with user types of similar names

macro_rules! make_ty_mod {
    ($modname:ident, $ty:tt) => {
        mod $modname {
            struct $ty {
                a: i32,
            }

            fn assignment() {
                let $ty = ();
            }

            fn access(a: $ty) -> i32 {
                a.a
            }
        }
    };
}

make_ty_mod!(check_f16, f16);
make_ty_mod!(check_f32, f32);
make_ty_mod!(check_f64, f64);
make_ty_mod!(check_f128, f128);

fn main() {}
