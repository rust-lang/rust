// Check that we reject type projections behind trait aliases that mention `Self`.
//
// The author of the trait object type can't fix this unlike the supertrait bound
// equivalent where they just need to explicitly specify the assoc type.

// issue: <https://github.com/rust-lang/rust/issues/139082>

#![feature(trait_alias)]

trait F = Fn() -> Self;

trait G = H<T = Self>;
trait H { type T: ?Sized; }

fn main() {
    let _: dyn F; //~ ERROR associated type binding in trait object type mentions `Self`
    let _: dyn G; //~ ERROR associated type binding in trait object type mentions `Self`
}
