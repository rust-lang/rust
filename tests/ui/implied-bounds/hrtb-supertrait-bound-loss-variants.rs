//@ compile-flags: -D hrtb_supertrait_lost_implied_bounds
// Regression test variants for #84591 — HRTB supertrait implied bound loss

// Variant A: Direct supertrait with dropped type param
trait SubA<'a, 'b, R>: SuperA<'a, 'b> {}
trait SuperA<'a, 'b> {
    fn convert(s: &'a str) -> &'b str;
}

fn variant_a<S>()
where
    S: for<'a, 'b> SubA<'a, 'b, &'b &'a ()>,
    //~^ ERROR hrtb_supertrait_lost_implied_bounds
    //~| WARN this was previously accepted by the compiler but is being phased out
{}

// Sound: no implied bounds lost (R doesn't reference bound lifetimes)
trait SubB<'a, 'b, R>: SuperB<'a, 'b> {}
trait SuperB<'a, 'b> {}

fn sound_variant<S>()
where
    S: for<'a, 'b> SubB<'a, 'b, i32>,  // i32 has no lifetime implications
{}

fn main() {}
