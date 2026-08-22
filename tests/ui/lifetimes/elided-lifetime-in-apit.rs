//! Regression test for <https://github.com/rust-lang/rust/issues/122903>
//!
//! Late resolver has a special case for `self` parameters where it looks for `&mut? Self`.
//! If that type appears inside an impl-trait, AST->HIR lowering could ICE bacause it cannot
//! find the definition of that lifetime.

struct Struct;

impl Struct {
    fn box_box_ref_Struct(
        self: impl FnMut(Box<impl FnMut(&mut Self)>),
        //~^ ERROR nested `impl Trait` is not allowed
        //~| ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds
        //~| ERROR invalid generic `self` parameter type: `impl FnMut(Box<impl FnMut(&mut Self)>)`
    ) -> &u32 {
        //~^ ERROR missing lifetime specifier
        &1
    }
}

fn main() {}
