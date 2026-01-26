// Check that we reject const projections behind trait aliases that mention `Self`.
// The code below is pretty artifical and contains a type mismatch anyway but we still need to
// reject it & lower the `Self` ty param to a `{type error}` to avoid ICEs down the line.
//
// The author of the trait object type can't fix this unlike the supertrait bound
// equivalent where they just need to explicitly specify the assoc const.

#![feature(min_generic_const_args, trait_alias)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const Y: i32;
}

struct Hold<T: ?Sized>(T);

trait Bound = Trait<Y = { Hold::<Self> }>;

fn main() {
    let _: dyn Bound; //~ ERROR associated constant binding in trait object type mentions `Self`
}
