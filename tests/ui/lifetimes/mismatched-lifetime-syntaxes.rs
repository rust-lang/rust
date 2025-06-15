#![deny(mismatched_lifetime_syntaxes)]

#[derive(Copy, Clone)]
struct ContainsLifetime<'a>(&'a u8);

struct S(u8);

fn explicit_bound_ref_to_implicit_ref<'a>(v: &'a u8) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn explicit_bound_ref_to_explicit_anonymous_ref<'a>(v: &'a u8) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

// ---

fn implicit_path_to_explicit_anonymous_path(v: ContainsLifetime) -> ContainsLifetime<'_> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn explicit_anonymous_path_to_implicit_path(v: ContainsLifetime<'_>) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn explicit_bound_path_to_implicit_path<'a>(v: ContainsLifetime<'a>) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn explicit_bound_path_to_explicit_anonymous_path<'a>(
    v: ContainsLifetime<'a>,
    //~^ ERROR lifetime flowing from input to output with different syntax
) -> ContainsLifetime<'_> {
    v
}

// ---

fn implicit_ref_to_implicit_path(v: &u8) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

fn explicit_anonymous_ref_to_implicit_path(v: &'_ u8) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

fn explicit_bound_ref_to_implicit_path<'a>(v: &'a u8) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

fn explicit_bound_ref_to_explicit_anonymous_path<'a>(v: &'a u8) -> ContainsLifetime<'_> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

// ---

fn implicit_path_to_implicit_ref(v: ContainsLifetime) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

fn implicit_path_to_explicit_anonymous_ref(v: ContainsLifetime) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

fn explicit_bound_path_to_implicit_ref<'a>(v: ContainsLifetime<'a>) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

fn explicit_bound_path_to_explicit_anonymous_ref<'a>(v: ContainsLifetime<'a>) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

impl S {
    fn method_explicit_bound_ref_to_implicit_ref<'a>(&'a self) -> &u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        &self.0
    }

    fn method_explicit_bound_ref_to_explicit_anonymous_ref<'a>(&'a self) -> &'_ u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        &self.0
    }

    // ---

    fn method_explicit_anonymous_ref_to_implicit_path(&'_ self) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }

    fn method_explicit_bound_ref_to_implicit_path<'a>(&'a self) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }

    fn method_explicit_bound_ref_to_explicit_anonymous_path<'a>(&'a self) -> ContainsLifetime<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }
}

// If a function uses the `'static` lifetime, we should not suggest
// replacing it with an explicitly anonymous or implicit
// lifetime. Only suggest using `'static` everywhere.
mod static_suggestions {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    struct S(u8);

    fn static_ref_to_implicit_ref(v: &'static u8) -> &u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        v
    }

    fn static_ref_to_explicit_anonymous_ref(v: &'static u8) -> &'_ u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        v
    }

    fn static_ref_to_implicit_path(v: &'static u8) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(v)
    }

    fn static_ref_to_explicit_anonymous_path(v: &'static u8) -> ContainsLifetime<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(v)
    }

    impl S {
        fn static_ref_to_implicit_ref(&'static self) -> &u8 {
            //~^ ERROR lifetime flowing from input to output with different syntax
            &self.0
        }

        fn static_ref_to_explicit_anonymous_ref(&'static self) -> &'_ u8 {
            //~^ ERROR lifetime flowing from input to output with different syntax
            &self.0
        }

        fn static_ref_to_implicit_path(&'static self) -> ContainsLifetime {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(&self.0)
        }

        fn static_ref_to_explicit_anonymous_path(&'static self) -> ContainsLifetime<'_> {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(&self.0)
        }
    }
}

/// `impl Trait` uses lifetimes in some additional ways.
mod impl_trait {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    fn explicit_bound_ref_to_impl_trait_bound<'a>(v: &'a u8) -> impl FnOnce() + '_ {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn explicit_bound_ref_to_impl_trait_precise_capture<'a>(v: &'a u8) -> impl FnOnce() + use<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn explicit_bound_path_to_impl_trait_bound<'a>(v: ContainsLifetime<'a>) -> impl FnOnce() + '_ {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn explicit_bound_path_to_impl_trait_precise_capture<'a>(
        v: ContainsLifetime<'a>,
        //~^ ERROR lifetime flowing from input to output with different syntax
    ) -> impl FnOnce() + use<'_> {
        move || _ = v
    }
}

/// `dyn Trait` uses lifetimes in some additional ways.
mod dyn_trait {
    use std::iter;

    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    fn explicit_bound_ref_to_dyn_trait_bound<'a>(v: &'a u8) -> Box<dyn Iterator<Item = &u8> + '_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        Box::new(iter::once(v))
    }

    fn explicit_bound_path_to_dyn_trait_bound<'a>(
        v: ContainsLifetime<'a>,
        //~^ ERROR lifetime flowing from input to output with different syntax
    ) -> Box<dyn Iterator<Item = ContainsLifetime> + '_> {
        Box::new(iter::once(v))
    }
}

/// These tests serve to exercise edge cases of the lint formatting
mod diagnostic_output {
    fn multiple_outputs<'a>(v: &'a u8) -> (&u8, &u8) {
        //~^ ERROR lifetime flowing from input to output with different syntax
        (v, v)
    }
}

/// Trait functions are represented differently in the HIR. Make sure
/// we visit them.
mod trait_functions {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    trait TheTrait {
        fn implicit_ref_to_implicit_path(v: &u8) -> ContainsLifetime;
        //~^ ERROR lifetime flowing from input to output with different syntax

        fn method_implicit_ref_to_implicit_path(&self) -> ContainsLifetime;
        //~^ ERROR lifetime flowing from input to output with different syntax
    }

    impl TheTrait for &u8 {
        fn implicit_ref_to_implicit_path(v: &u8) -> ContainsLifetime {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(v)
        }

        fn method_implicit_ref_to_implicit_path(&self) -> ContainsLifetime {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(self)
        }
    }
}

/// Extern functions are represented differently in the HIR. Make sure
/// we visit them.
mod foreign_functions {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    extern "Rust" {
        fn implicit_ref_to_implicit_path(v: &u8) -> ContainsLifetime;
        //~^ ERROR lifetime flowing from input to output with different syntax
    }
}

/// These usages are expected to **not** trigger the lint
mod acceptable_uses {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    struct S(u8);

    fn implicit_ref_to_implicit_ref(v: &u8) -> &u8 {
        v
    }

    fn explicit_anonymous_ref_to_explicit_anonymous_ref(v: &'_ u8) -> &'_ u8 {
        v
    }

    fn explicit_bound_ref_to_explicit_bound_ref<'a>(v: &'a u8) -> &'a u8 {
        v
    }

    fn implicit_path_to_implicit_path(v: ContainsLifetime) -> ContainsLifetime {
        v
    }

    fn explicit_anonymous_path_to_explicit_anonymous_path(
        v: ContainsLifetime<'_>,
    ) -> ContainsLifetime<'_> {
        v
    }

    fn explicit_bound_path_to_explicit_bound_path<'a>(
        v: ContainsLifetime<'a>,
    ) -> ContainsLifetime<'a> {
        v
    }

    fn explicit_anonymous_ref_to_explicit_anonymous_path(v: &'_ u8) -> ContainsLifetime<'_> {
        ContainsLifetime(v)
    }

    fn explicit_bound_ref_to_explicit_bound_path<'a>(v: &'a u8) -> ContainsLifetime<'a> {
        ContainsLifetime(v)
    }

    fn explicit_anonymous_path_to_explicit_anonymous_ref(v: ContainsLifetime<'_>) -> &'_ u8 {
        v.0
    }

    fn explicit_bound_path_to_explicit_bound_ref<'a>(v: ContainsLifetime<'a>) -> &'a u8 {
        v.0
    }

    // These may be surprising, but ampersands count as enough of a
    // visual indicator that a reference exists that we treat
    // references with implicit lifetimes the same as if they were
    // explicitly anonymous.
    fn implicit_ref_to_explicit_anonymous_ref(v: &u8) -> &'_ u8 {
        v
    }

    fn explicit_anonymous_ref_to_implicit_ref(v: &'_ u8) -> &u8 {
        v
    }

    fn implicit_ref_to_explicit_anonymous_path(v: &u8) -> ContainsLifetime<'_> {
        ContainsLifetime(v)
    }

    fn explicit_anonymous_path_to_implicit_ref(v: ContainsLifetime<'_>) -> &u8 {
        v.0
    }

    impl S {
        fn method_implicit_ref_to_explicit_anonymous_ref(&self) -> &'_ u8 {
            &self.0
        }

        fn method_explicit_anonymous_ref_to_implicit_ref(&'_ self) -> &u8 {
            &self.0
        }

        fn method_implicit_ref_to_explicit_anonymous_path(&self) -> ContainsLifetime<'_> {
            ContainsLifetime(&self.0)
        }
    }

    // `dyn Trait` has an "embedded" lifetime that we should **not**
    // lint about.
    fn dyn_trait_does_not_have_a_lifetime_generic(v: &u8) -> &dyn core::fmt::Debug {
        v
    }
}

fn main() {}
