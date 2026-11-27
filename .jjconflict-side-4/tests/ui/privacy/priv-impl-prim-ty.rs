//@ run-pass
//@ aux-build:priv-impl-prim-ty.rs


extern crate priv_impl_prim_ty as bar;

pub fn main() {
    bar::frob(1);

}
