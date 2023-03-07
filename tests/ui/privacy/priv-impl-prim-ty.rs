// run-pass
// aux-build:priv-impl-prim-ty.rs

// pretty-expanded FIXME #23616

extern crate priv_impl_prim_ty as bar;

pub fn main() {
    bar::frob(1);

}
