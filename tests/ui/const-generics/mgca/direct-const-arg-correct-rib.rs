//! make sure TyKind::DirectConstArg resolves properly with the correct ribs and doesn't ICE
#![feature(min_generic_const_args)]

struct S;
struct V<const N: usize>;
impl S {
    fn f(self) {
        let _: V<core::direct_const_arg!(self)>;
        //~^ ERROR attempt to use a non-constant value in a constant
    }
}

fn main() {}
