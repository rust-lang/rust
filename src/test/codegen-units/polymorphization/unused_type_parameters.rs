// compile-flags:-Zpolymorphize=on -Zprint-mono-items=lazy -Copt-level=1
// ignore-tidy-linelength

#![crate_type = "rlib"]

// This test checks that the polymorphization analysis correctly reduces the
// generated mono items.

mod functions {
    // Function doesn't have any type parameters to be unused.
    pub fn no_parameters() {}

//~ MONO_ITEM fn unused_type_parameters::functions[0]::no_parameters[0]

    // Function has an unused type parameter.
    pub fn unused<T>() {
    }

//~ MONO_ITEM fn unused_type_parameters::functions[0]::unused[0]<T>

    // Function uses type parameter in value of a binding.
    pub fn used_binding_value<T: Default>() {
        let _: T = Default::default();
    }

//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_binding_value[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_binding_value[0]<u64>

    // Function uses type parameter in type of a binding.
    pub fn used_binding_type<T>() {
        let _: Option<T> = None;
    }

//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_binding_type[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_binding_type[0]<u64>

    // Function uses type parameter in argument.
    pub fn used_argument<T>(_: T) {
    }

//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_argument[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_argument[0]<u64>
//
    // Function uses type parameter in substitutions to another function.
    pub fn used_substs<T>() {
        unused::<T>()
    }

//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_substs[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::functions[0]::used_substs[0]<u64>
}


mod closures {
    // Function doesn't have any type parameters to be unused.
    pub fn no_parameters() {
        let _ = || {};
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::no_parameters[0]

    // Function has an unused type parameter in parent and closure.
    pub fn unused<T>() -> u32 {
        let add_one = |x: u32| x + 1;
        add_one(3)
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::unused[0]::{{closure}}[0]<T, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::unused[0]<T>

    // Function has an unused type parameter in closure, but not in parent.
    pub fn used_parent<T: Default>() -> u32 {
        let _: T = Default::default();
        let add_one = |x: u32| x + 1;
        add_one(3)
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_parent[0]::{{closure}}[0]<T, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_parent[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_parent[0]<u64>

    // Function uses type parameter in value of a binding in closure.
    pub fn used_binding_value<T: Default>() -> T {
        let x = || {
            let y: T = Default::default();
            y
        };

        x()
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_value[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn(()) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_value[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn(()) -> u64, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_value[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_value[0]<u64>

    // Function uses type parameter in type of a binding in closure.
    pub fn used_binding_type<T>() -> Option<T> {
        let x = || {
            let y: Option<T> = None;
            y
        };

        x()
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_type[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn(()) -> core::option[0]::Option[0]<u32>, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_type[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn(()) -> core::option[0]::Option[0]<u64>, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_type[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_binding_type[0]<u64>

    // Function and closure uses type parameter in argument.
    pub fn used_argument<T>(t: T) -> u32 {
        let x = |_: T| 3;
        x(t)
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn((u64)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument[0]<u64>

    // Closure uses type parameter in argument.
    pub fn used_argument_closure<T: Default>() -> u32 {
        let t: T = Default::default();
        let x = |_: T| 3;
        x(t)
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument_closure[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument_closure[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn((u64)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument_closure[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_argument_closure[0]<u64>

    // Closure uses type parameter as upvar.
    pub fn used_upvar<T: Default>() -> T {
        let x: T = Default::default();
        let y = || x;
        y()
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_upvar[0]::{{closure}}[0]<u32, i32, extern "rust-call" fn(()) -> u32, (u32)>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_upvar[0]::{{closure}}[0]<u64, i32, extern "rust-call" fn(()) -> u64, (u64)>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_upvar[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_upvar[0]<u64>

    // Closure uses type parameter in substitutions to another function.
    pub fn used_substs<T>() {
        let x = || super::functions::unused::<T>();
        x()
    }

//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_substs[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn(()), ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_substs[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn(()), ()>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_substs[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::closures[0]::used_substs[0]<u64>
}

mod methods {
    pub struct Foo<F>(F);

    impl<F: Default> Foo<F> {
        // Function has an unused type parameter from impl.
        pub fn unused_impl() {
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::unused_impl[0]<F>

        // Function has an unused type parameter from impl and fn.
        pub fn unused_both<G: Default>() {
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::unused_both[0]<F, G>

        // Function uses type parameter from impl.
        pub fn used_impl() {
            let _: F = Default::default();
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_impl[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_impl[0]<u64>

        // Function uses type parameter from impl.
        pub fn used_fn<G: Default>() {
            let _: G = Default::default();
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_fn[0]<F, u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_fn[0]<F, u64>

        // Function uses type parameter from impl.
        pub fn used_both<G: Default>() {
            let _: F = Default::default();
            let _: G = Default::default();
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_both[0]<u32, u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_both[0]<u64, u64>

        // Function uses type parameter in substitutions to another function.
        pub fn used_substs() {
            super::functions::unused::<F>()
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_substs[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::used_substs[0]<u64>

        // Function has an unused type parameter from impl and fn.
        pub fn closure_unused_all<G: Default>() -> u32 {
            let add_one = |x: u32| x + 1;
            add_one(3)
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_unused_all[0]::{{closure}}[0]<F, G, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_unused_all[0]<F, G>

        // Function uses type parameter from impl and fn in closure.
        pub fn closure_used_both<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: F = Default::default();
                let _: G = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_both[0]::{{closure}}[0]<u32, u32, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_both[0]::{{closure}}[0]<u64, u64, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_both[0]<u32, u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_both[0]<u64, u64>

        // Function uses type parameter from fn in closure.
        pub fn closure_used_fn<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: G = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_fn[0]::{{closure}}[0]<F, u32, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_fn[0]::{{closure}}[0]<F, u64, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_fn[0]<F, u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_fn[0]<F, u64>

        // Function uses type parameter from impl in closure.
        pub fn closure_used_impl<G: Default>() -> u32 {
            let add_one = |x: u32| {
                let _: F = Default::default();
                x + 1
            };

            add_one(3)
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_impl[0]::{{closure}}[0]<u32, G, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_impl[0]::{{closure}}[0]<u64, G, i8, extern "rust-call" fn((u32)) -> u32, ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_impl[0]<u32, G>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_impl[0]<u64, G>

        // Closure uses type parameter in substitutions to another function.
        pub fn closure_used_substs() {
            let x = || super::functions::unused::<F>();
            x()
        }

//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_substs[0]::{{closure}}[0]<u32, i8, extern "rust-call" fn(()), ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_substs[0]::{{closure}}[0]<u64, i8, extern "rust-call" fn(()), ()>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_substs[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::methods[0]::{{impl}}[0]::closure_used_substs[0]<u64>
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

//~ MONO_ITEM fn unused_type_parameters::dispatch[0]<u32>
//~ MONO_ITEM fn unused_type_parameters::dispatch[0]<u64>

pub fn foo() {
    // Generate two copies of each function to check that where the type parameter is unused,
    // there is only a single copy.
    dispatch::<u32>();
    dispatch::<u64>();
}

//~ MONO_ITEM fn unused_type_parameters::foo[0] @@ unused_type_parameters-cgu.0[External]

// These are all the items that aren't relevant to the test.
//~ MONO_ITEM fn core::default[0]::{{impl}}[6]::default[0]
//~ MONO_ITEM fn core::default[0]::{{impl}}[7]::default[0]
