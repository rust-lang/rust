#![deny(mismatched_lifetime_syntaxes)]

// `elided_named_lifetimes` overlaps with `lifetime_style_mismatch`, ignore it for now
#![allow(elided_named_lifetimes)]

#[derive(Copy, Clone)]
struct ContainsLifetime<'a>(&'a u8);

struct S(u8);

fn named_ref_to_hidden_ref<'a>(v: &'a u8) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn named_ref_to_anonymous_ref<'a>(v: &'a u8) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

// ---

fn hidden_path_to_anonymous_path(v: ContainsLifetime) -> ContainsLifetime<'_> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn anonymous_path_to_hidden_path(v: ContainsLifetime<'_>) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn named_path_to_hidden_path<'a>(v: ContainsLifetime<'a>) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

fn named_path_to_anonymous_path<'a>(v: ContainsLifetime<'a>) -> ContainsLifetime<'_> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v
}

// ---

fn anonymous_ref_to_hidden_path(v: &'_ u8) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

fn named_ref_to_hidden_path<'a>(v: &'a u8) -> ContainsLifetime {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

fn named_ref_to_anonymous_path<'a>(v: &'a u8) -> ContainsLifetime<'_> {
    //~^ ERROR lifetime flowing from input to output with different syntax
    ContainsLifetime(v)
}

// ---

fn hidden_path_to_anonymous_ref(v: ContainsLifetime) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

fn named_path_to_hidden_ref<'a>(v: ContainsLifetime<'a>) -> &u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

fn named_path_to_anonymous_ref<'a>(v: ContainsLifetime<'a>) -> &'_ u8 {
    //~^ ERROR lifetime flowing from input to output with different syntax
    v.0
}

impl S {
    fn method_named_ref_to_hidden_ref<'a>(&'a self) -> &u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        &self.0
    }

    fn method_named_ref_to_anonymous_ref<'a>(&'a self) -> &'_ u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        &self.0
    }

    // ---

    fn method_anonymous_ref_to_hidden_path(&'_ self) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }

    fn method_named_ref_to_hidden_path<'a>(&'a self) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }

    fn method_named_ref_to_anonymous_path<'a>(&'a self) -> ContainsLifetime<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(&self.0)
    }
}

// If a function uses the `'static` named lifetime, we should not
// suggest replacing it with an anonymous or hidden lifetime. Only
// suggest using `'static` everywhere.
mod static_suggestions {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    struct S(u8);

    fn static_ref_to_hidden_ref(v: &'static u8) -> &u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        v
    }

    fn static_ref_to_anonymous_ref(v: &'static u8) -> &'_ u8 {
        //~^ ERROR lifetime flowing from input to output with different syntax
        v
    }

    fn static_ref_to_hidden_path(v: &'static u8) -> ContainsLifetime {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(v)
    }

    fn static_ref_to_anonymous_path(v: &'static u8) -> ContainsLifetime<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        ContainsLifetime(v)
    }

    impl S {
        fn static_ref_to_hidden_ref(&'static self) -> &u8 {
            //~^ ERROR lifetime flowing from input to output with different syntax
            &self.0
        }

        fn static_ref_to_anonymous_ref(&'static self) -> &'_ u8 {
            //~^ ERROR lifetime flowing from input to output with different syntax
            &self.0
        }

        fn static_ref_to_hidden_path(&'static self) -> ContainsLifetime {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(&self.0)
        }

        fn static_ref_to_anonymous_path(&'static self) -> ContainsLifetime<'_> {
            //~^ ERROR lifetime flowing from input to output with different syntax
            ContainsLifetime(&self.0)
        }
    }
}

/// `impl Trait` uses lifetimes in some additional ways.
mod impl_trait {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    fn named_ref_to_impl_trait_bound<'a>(v: &'a u8) -> impl FnOnce() + '_ {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn named_ref_to_impl_trait_precise_capture<'a>(v: &'a u8) -> impl FnOnce() + use<'_> {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn named_path_to_impl_trait_bound<'a>(v: ContainsLifetime<'a>) -> impl FnOnce() + '_ {
        //~^ ERROR lifetime flowing from input to output with different syntax
        move || _ = v
    }

    fn named_path_to_impl_trait_precise_capture<'a>(
        v: ContainsLifetime<'a>,
        //~^ ERROR lifetime flowing from input to output with different syntax
    ) -> impl FnOnce() + use<'_> {
        move || _ = v
    }
}

/// These tests serve to exercise edge cases of the lint formatting
mod diagnostic_output {
    fn multiple_outputs<'a>(v: &'a u8) -> (&u8, &u8) {
        //~^ ERROR lifetime flowing from input to output with different syntax
        (v, v)
    }
}

/// These usages are expected to **not** trigger the lint
mod acceptable_uses {
    #[derive(Copy, Clone)]
    struct ContainsLifetime<'a>(&'a u8);

    struct S(u8);

    fn hidden_ref_to_hidden_ref(v: &u8) -> &u8 {
        v
    }

    fn anonymous_ref_to_anonymous_ref(v: &'_ u8) -> &'_ u8 {
        v
    }

    fn named_ref_to_named_ref<'a>(v: &'a u8) -> &'a u8 {
        v
    }

    fn hidden_path_to_hidden_path(v: ContainsLifetime) -> ContainsLifetime {
        v
    }

    fn anonymous_path_to_anonymous_path(v: ContainsLifetime<'_>) -> ContainsLifetime<'_> {
        v
    }

    fn named_path_to_named_path<'a>(v: ContainsLifetime<'a>) -> ContainsLifetime<'a> {
        v
    }

    fn hidden_ref_to_hidden_path(v: &u8) -> ContainsLifetime {
        ContainsLifetime(v)
    }

    fn anonymous_ref_to_anonymous_path(v: &'_ u8) -> ContainsLifetime<'_> {
        ContainsLifetime(v)
    }

    fn named_ref_to_named_path<'a>(v: &'a u8) -> ContainsLifetime<'a> {
        ContainsLifetime(v)
    }

    fn hidden_path_to_hidden_ref(v: ContainsLifetime) -> &u8 {
        v.0
    }

    fn anonymous_path_to_anonymous_ref(v: ContainsLifetime<'_>) -> &'_ u8 {
        v.0
    }

    fn named_path_to_named_ref<'a>(v: ContainsLifetime<'a>) -> &'a u8 {
        v.0
    }

    // These may be surprising, but ampersands count as enough of a
    // visual indicator that a reference exists that we treat
    // references with hidden lifetimes the same as if they were
    // anonymous.
    fn hidden_ref_to_anonymous_ref(v: &u8) -> &'_ u8 {
        v
    }

    fn anonymous_ref_to_hidden_ref(v: &'_ u8) -> &u8 {
        v
    }

    fn hidden_ref_to_anonymous_path(v: &u8) -> ContainsLifetime<'_> {
        ContainsLifetime(v)
    }

    fn anonymous_path_to_hidden_ref(v: ContainsLifetime<'_>) -> &u8 {
        v.0
    }

    impl S {
        fn method_hidden_ref_to_anonymous_ref(&self) -> &'_ u8 {
            &self.0
        }

        fn method_anonymous_ref_to_hidden_ref(&'_ self) -> &u8 {
            &self.0
        }

        fn method_hidden_ref_to_anonymous_path(&self) -> ContainsLifetime<'_> {
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
