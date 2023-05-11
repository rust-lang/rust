// build-fail
// compile-flags:-Zpolymorphize=on
#![feature(stmt_expr_attributes, rustc_attrs)]

// This test checks that the polymorphization analysis correctly detects unused type
// parameters in closures.

// Function doesn't have any generic parameters to be unused.
#[rustc_polymorphize_error]
pub fn no_parameters() {
    let _ = || {};
}

// Function has an unused generic parameter in parent and closure.
#[rustc_polymorphize_error]
pub fn unused<T>() -> u32 {
    //~^ ERROR item has unused generic parameters

    let add_one = |x: u32| x + 1;
    //~^ ERROR item has unused generic parameters
    add_one(3)
}

// Function has an unused generic parameter in closure, but not in parent.
#[rustc_polymorphize_error]
pub fn used_parent<T: Default>() -> u32 {
    let _: T = Default::default();
    let add_one = |x: u32| x + 1;
    //~^ ERROR item has unused generic parameters
    add_one(3)
}

// Function uses generic parameter in value of a binding in closure.
#[rustc_polymorphize_error]
pub fn used_binding_value<T: Default>() -> T {
    let x = || {
        let y: T = Default::default();
        y
    };

    x()
}

// Function uses generic parameter in generic of a binding in closure.
#[rustc_polymorphize_error]
pub fn used_binding_generic<T>() -> Option<T> {
    let x = || {
        let y: Option<T> = None;
        y
    };

    x()
}

// Function and closure uses generic parameter in argument.
#[rustc_polymorphize_error]
pub fn used_argument<T>(t: T) -> u32 {
    let x = |_: T| 3;
    x(t)
}

// Closure uses generic parameter in argument.
#[rustc_polymorphize_error]
pub fn used_argument_closure<T: Default>() -> u32 {
    let t: T = Default::default();

    let x = |_: T| 3;
    x(t)
}

// Closure uses generic parameter as upvar.
#[rustc_polymorphize_error]
pub fn used_upvar<T: Default>() -> T {
    let x: T = Default::default();

    let y = || x;
    y()
}

// Closure uses generic parameter in substitutions to another function.
#[rustc_polymorphize_error]
pub fn used_substs<T>() -> u32 {
    let x = || unused::<T>();
    x()
}

struct Foo<F>(F);

impl<F: Default> Foo<F> {
    // Function has an unused generic parameter from impl and fn.
    #[rustc_polymorphize_error]
    pub fn unused_all<G: Default>() -> u32 {
        //~^ ERROR item has unused generic parameters
        let add_one = |x: u32| x + 1;
        //~^ ERROR item has unused generic parameters
        add_one(3)
    }

    // Function uses generic parameter from impl and fn in closure.
    #[rustc_polymorphize_error]
    pub fn used_both<G: Default>() -> u32 {
        let add_one = |x: u32| {
            let _: F = Default::default();
            let _: G = Default::default();
            x + 1
        };

        add_one(3)
    }

    // Function uses generic parameter from fn in closure.
    #[rustc_polymorphize_error]
    pub fn used_fn<G: Default>() -> u32 {
        //~^ ERROR item has unused generic parameters
        let add_one = |x: u32| {
            //~^ ERROR item has unused generic parameters
            let _: G = Default::default();
            x + 1
        };

        add_one(3)
    }

    // Function uses generic parameter from impl in closure.
    #[rustc_polymorphize_error]
    pub fn used_impl<G: Default>() -> u32 {
        //~^ ERROR item has unused generic parameters
        let add_one = |x: u32| {
            //~^ ERROR item has unused generic parameters
            let _: F = Default::default();
            x + 1
        };

        add_one(3)
    }

    // Closure uses generic parameter in substitutions to another function.
    #[rustc_polymorphize_error]
    pub fn used_substs() -> u32 {
        let x = || unused::<F>();
        x()
    }
}

fn main() {
    no_parameters();
    let _ = unused::<u32>();
    let _ = used_parent::<u32>();
    let _ = used_binding_value::<u32>();
    let _ = used_binding_generic::<u32>();
    let _ = used_argument(3u32);
    let _ = used_argument_closure::<u32>();
    let _ = used_upvar::<u32>();
    let _ = used_substs::<u32>();

    let _ = Foo::<u32>::unused_all::<u32>();
    let _ = Foo::<u32>::used_both::<u32>();
    let _ = Foo::<u32>::used_impl::<u32>();
    let _ = Foo::<u32>::used_fn::<u32>();
    let _ = Foo::<u32>::used_substs();
}
