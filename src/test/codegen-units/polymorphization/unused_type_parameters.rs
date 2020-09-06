// compile-flags:-Zpolymorphize=on -Zprint-mono-items=lazy -Copt-level=1

#![crate_type = "rlib"]

// This test checks that the polymorphization analysis correctly reduces the
// generated mono items.

mod functions {
    // Function doesn't have any type parameters to be unused.
    pub fn no_parameters() {}

//~ MONO_ITEM fn functions::no_parameters

    // Function has an unused type parameter.
    pub fn unused<T>() {
    }

//~ MONO_ITEM fn functions::unused::<T>

    // Function uses type parameter in value of a binding.
    pub fn used_binding_value<T: Default>() {
        let _: T = Default::default();
    }

//~ MONO_ITEM fn functions::used_binding_value::<u32>
//~ MONO_ITEM fn functions::used_binding_value::<u64>

    // Function uses type parameter in type of a binding.
    pub fn used_binding_type<T>() {
        let _: Option<T> = None;
    }

//~ MONO_ITEM fn functions::used_binding_type::<u32>
//~ MONO_ITEM fn functions::used_binding_type::<u64>

    // Function uses type parameter in argument.
    pub fn used_argument<T>(_: T) {
    }

//~ MONO_ITEM fn functions::used_argument::<u32>
//~ MONO_ITEM fn functions::used_argument::<u64>
//
    // Function uses type parameter in substitutions to another function.
    pub fn used_substs<T>() {
        unused::<T>()
    }

//~ MONO_ITEM fn functions::used_substs::<u32>
//~ MONO_ITEM fn functions::used_substs::<u64>
}


mod closures {
    // Function doesn't have any type parameters to be unused.
    pub fn no_parameters() {
        let _ = || {};
    }

//~ MONO_ITEM fn closures::no_parameters

    // Function has an unused type parameter in parent and closure.
    pub fn unused<T>() -> u32 {
        let add_one = |x: u32| x + 1;
        add_one(3)
    }

//~ MONO_ITEM fn closures::unused::<T>::{{closure}}#0
//~ MONO_ITEM fn closures::unused::<T>

    // Function has an unused type parameter in closure, but not in parent.
    pub fn used_parent<T: Default>() -> u32 {
        let _: T = Default::default();
        let add_one = |x: u32| x + 1;
        add_one(3)
    }

//~ MONO_ITEM fn closures::used_parent::<T>::{{closure}}#0
//~ MONO_ITEM fn closures::used_parent::<u32>
//~ MONO_ITEM fn closures::used_parent::<u64>

    // Function uses type parameter in value of a binding in closure.
    pub fn used_binding_value<T: Default>() -> T {
        let x = || {
            let y: T = Default::default();
            y
        };

        x()
    }

//~ MONO_ITEM fn closures::used_binding_value::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_binding_value::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_binding_value::<u32>
//~ MONO_ITEM fn closures::used_binding_value::<u64>

    // Function uses type parameter in type of a binding in closure.
    pub fn used_binding_type<T>() -> Option<T> {
        let x = || {
            let y: Option<T> = None;
            y
        };

        x()
    }

//~ MONO_ITEM fn closures::used_binding_type::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_binding_type::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_binding_type::<u32>
//~ MONO_ITEM fn closures::used_binding_type::<u64>

    // Function and closure uses type parameter in argument.
    pub fn used_argument<T>(t: T) -> u32 {
        let x = |_: T| 3;
        x(t)
    }

//~ MONO_ITEM fn closures::used_argument::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_argument::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_argument::<u32>
//~ MONO_ITEM fn closures::used_argument::<u64>

    // Closure uses type parameter in argument.
    pub fn used_argument_closure<T: Default>() -> u32 {
        let t: T = Default::default();
        let x = |_: T| 3;
        x(t)
    }

//~ MONO_ITEM fn closures::used_argument_closure::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_argument_closure::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_argument_closure::<u32>
//~ MONO_ITEM fn closures::used_argument_closure::<u64>

    // Closure uses type parameter as upvar.
    pub fn used_upvar<T: Default>() -> T {
        let x: T = Default::default();
        let y = || x;
        y()
    }

//~ MONO_ITEM fn closures::used_upvar::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_upvar::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_upvar::<u32>
//~ MONO_ITEM fn closures::used_upvar::<u64>

    // Closure uses type parameter in substitutions to another function.
    pub fn used_substs<T>() {
        let x = || super::functions::unused::<T>();
        x()
    }

//~ MONO_ITEM fn closures::used_substs::<u32>::{{closure}}#0
//~ MONO_ITEM fn closures::used_substs::<u64>::{{closure}}#0
//~ MONO_ITEM fn closures::used_substs::<u32>
//~ MONO_ITEM fn closures::used_substs::<u64>
}

mod methods {
    pub struct Foo<F>(F);

    impl<F: Default> Foo<F> {
        // Function has an unused type parameter from impl.
        pub fn unused_impl() {
        }

//~ MONO_ITEM fn methods::Foo::<F>::unused_impl

        // Function has an unused type parameter from impl and fn.
        pub fn unused_both<G: Default>() {
        }

//~ MONO_ITEM fn methods::Foo::<F>::unused_both::<G>

        // Function uses type parameter from impl.
        pub fn used_impl() {
            let _: F = Default::default();
        }

//~ MONO_ITEM fn methods::Foo::<u32>::used_impl
//~ MONO_ITEM fn methods::Foo::<u64>::used_impl

        // Function uses type parameter from impl.
        pub fn used_fn<G: Default>() {
            let _: G = Default::default();
        }

//~ MONO_ITEM fn methods::Foo::<F>::used_fn::<u32>
//~ MONO_ITEM fn methods::Foo::<F>::used_fn::<u64>

        // Function uses type parameter from impl.
        pub fn used_both<G: Default>() {
            let _: F = Default::default();
            let _: G = Default::default();
        }

//~ MONO_ITEM fn methods::Foo::<u32>::used_both::<u32>
//~ MONO_ITEM fn methods::Foo::<u64>::used_both::<u64>

        // Function uses type parameter in substitutions to another function.
        pub fn used_substs() {
            super::functions::unused::<F>()
        }

//~ MONO_ITEM fn methods::Foo::<u32>::used_substs
//~ MONO_ITEM fn methods::Foo::<u64>::used_substs

        // Function has an unused type parameter from impl and fn.
        pub fn closure_unused_all<G: Default>() -> u32 {
            let add_one = |x: u32| x + 1;
            add_one(3)
        }

//~ MONO_ITEM fn methods::Foo::<F>::closure_unused_all::<G>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<F>::closure_unused_all::<G>

        // Function uses type parameter from impl and fn in closure.
        pub fn closure_used_both<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: F = Default::default();
                let _: G = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_both::<u32>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_both::<u64>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_both::<u32>
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_both::<u64>

        // Function uses type parameter from fn in closure.
        pub fn closure_used_fn<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: G = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn methods::Foo::<F>::closure_used_fn::<u32>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<F>::closure_used_fn::<u64>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<F>::closure_used_fn::<u32>
//~ MONO_ITEM fn methods::Foo::<F>::closure_used_fn::<u64>

        // Function uses type parameter from impl in closure.
        pub fn closure_used_impl<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: F = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_impl::<G>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_impl::<G>::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_impl::<G>
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_impl::<G>

        // Closure uses type parameter in substitutions to another function.
        pub fn closure_used_substs() {
            let x = || super::functions::unused::<F>();
            x()
        }

//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_substs::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_substs::{{closure}}#0
//~ MONO_ITEM fn methods::Foo::<u32>::closure_used_substs
//~ MONO_ITEM fn methods::Foo::<u64>::closure_used_substs
    }
}



fn dispatch<T: Default>() {
    functions::no_parameters();
    functions::unused::<T>();
    functions::used_binding_value::<T>();
    functions::used_binding_type::<T>();
    functions::used_argument::<T>(Default::default());
    functions::used_substs::<T>();

    closures::no_parameters();
    let _ = closures::unused::<T>();
    let _ = closures::used_parent::<T>();
    let _ = closures::used_binding_value::<T>();
    let _ = closures::used_binding_type::<T>();
    let _ = closures::used_argument::<T>(Default::default());
    let _ = closures::used_argument_closure::<T>();
    let _ = closures::used_upvar::<T>();
    let _ = closures::used_substs::<T>();

    methods::Foo::<T>::unused_impl();
    methods::Foo::<T>::unused_both::<T>();
    methods::Foo::<T>::used_impl();
    methods::Foo::<T>::used_fn::<T>();
    methods::Foo::<T>::used_both::<T>();
    methods::Foo::<T>::used_substs();
    let _ = methods::Foo::<T>::closure_unused_all::<T>();
    let _ = methods::Foo::<T>::closure_used_both::<T>();
    let _ = methods::Foo::<T>::closure_used_impl::<T>();
    let _ = methods::Foo::<T>::closure_used_fn::<T>();
    let _ = methods::Foo::<T>::closure_used_substs();
}

//~ MONO_ITEM fn dispatch::<u32>
//~ MONO_ITEM fn dispatch::<u64>

pub fn foo() {
    // Generate two copies of each function to check that where the type parameter is unused,
    // there is only a single copy.
    dispatch::<u32>();
    dispatch::<u64>();
}

//~ MONO_ITEM fn foo @@ unused_type_parameters-cgu.0[External]

// These are all the items that aren't relevant to the test.
//~ MONO_ITEM fn <u32 as std::default::Default>::default
//~ MONO_ITEM fn <u64 as std::default::Default>::default
