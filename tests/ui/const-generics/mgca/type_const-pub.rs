//@ check-pass
// This previously caused an ICE when checking reachability of a pub const item
// This is because reachability also tried to evaluate the #[type_const] which
// requires the item have a body. #[type_const] do not have bodies.
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

#[type_const]
pub const TYPE_CONST : usize = 1;
fn main() {
    print!("{}", TYPE_CONST)
}
