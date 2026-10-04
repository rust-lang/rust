// Check that a projection `Self::V` in a trait implementation,
// with an associated type named `V`, for an `enum` with a variant named `V`,
// results in triggering the deny-by-default lint `ambiguous_associated_items`.
// The lint suggests that qualified syntax should be used instead.
// That is, the user would write `<Self as Tr>::V`.
//
// The rationale for this is that while `enum` variants do currently
// not exist in the type namespace but solely in the value namespace,
// RFC #2593 "Enum variant types", would add enum variants to the type namespace.
// However, currently `enum` variants are resolved with high priority as
// they are resolved as inherent associated items.
// Should #2953 therefore be implemented, `Self::V` would suddenly switch
// from referring to the associated type `V` instead of the variant `V`.
// The lint exists to keep us forward compatible with #2593.
//
// As a closing note, provided that #2933 was implemented and
// if `enum` variants were given lower priority than associated types,
// it would be impossible to refer to the `enum` variant `V` whereas
// the associated type could be referred to with qualified syntax as seen above.

enum E {
    V
}

trait Tr {
    type V;
    fn f() -> Self::V;
}

impl Tr for E {
    type V = u8;
    fn f() -> Self::V { 0 }
    //~^ ERROR ambiguous associated item
    //~| WARN this was previously accepted
    //~| HELP use fully-qualified syntax
}

fn main() {}
