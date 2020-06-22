// build-fail
// compile-flags: -Zpolymorphize-errors

// This test checks that the polymorphization analysis correctly detects unused type
// parameters in functions.

// Function doesn't have any generic parameters to be unused.
pub fn no_parameters() {}

// Function has an unused generic parameter.
pub fn unused<T>() {
//~^ ERROR item has unused generic parameters
}

// Function uses generic parameter in value of a binding.
pub fn used_binding_value<T: Default>() {
    let _: T = Default::default();
}

// Function uses generic parameter in generic of a binding.
pub fn used_binding_generic<T>() {
    let _: Option<T> = None;
}

// Function uses generic parameter in argument.
pub fn used_argument<T>(_: T) {
}

// Function uses generic parameter in substitutions to another function.
pub fn used_substs<T>() {
    unused::<T>()
}

struct Foo<F>(F);

impl<F: Default> Foo<F> {
    // Function has an unused generic parameter from impl.
    pub fn unused_impl() {
//~^ ERROR item has unused generic parameters
    }

    // Function has an unused generic parameter from impl and fn.
    pub fn unused_both<G: Default>() {
//~^ ERROR item has unused generic parameters
    }

    // Function uses generic parameter from impl.
    pub fn used_impl() {
        let _: F = Default::default();
    }

    // Function uses generic parameter from impl.
    pub fn used_fn<G: Default>() {
//~^ ERROR item has unused generic parameters
        let _: G = Default::default();
    }

    // Function uses generic parameter from impl.
    pub fn used_both<G: Default>() {
        let _: F = Default::default();
        let _: G = Default::default();
    }

    // Function uses generic parameter in substitutions to another function.
    pub fn used_substs() {
        unused::<F>()
    }
}

fn main() {
    no_parameters();
    unused::<u32>();
    used_binding_value::<u32>();
    used_binding_generic::<u32>();
    used_argument(3u32);
    used_substs::<u32>();

    Foo::<u32>::unused_impl();
    Foo::<u32>::unused_both::<u32>();
    Foo::<u32>::used_impl();
    Foo::<u32>::used_fn::<u32>();
    Foo::<u32>::used_both::<u32>();
    Foo::<u32>::used_substs();
}
