//@ compile-flags: -Znext-solver -Zassumptions-on-binders

#![crate_type = "lib"]

// Same concept as `type_relation_binders_inside_solver-1.rs`, this time the type relation
// occuring when checking `for<'b> fn(&'b ()): Trait` holds and we have to equate the two
// higher ranked function pointer types.

trait Trait { }

fn req_trait<T: Trait>(_: T) { }

fn mk() -> for<'b> fn(&'b ()) { loop {} }

fn ice()
where
    (for<'b> fn(&'b ())): Trait
    //~^ ERROR the trait bound `for<'b> fn(&'b ()): Trait` is not satisfied
{
    req_trait(mk());
}
