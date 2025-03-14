//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// Regression test for trait-system-refactor-initiative#106. We previously
// tried to define other opaques while checking that opaques are well-formed.
//
// This resulted in undesirable ambiguity

#![feature(type_alias_impl_trait)]

mod ex0 {
    fn foo() -> (impl Sized, impl Sized) {
        ((), ())
    }
}
mod ex1 {
    type Tait1 = impl Sized;
    //[current]~^ ERROR unconstrained opaque type
    #[define_opaque(Tait1)]
    fn foo(x: Tait1) -> impl Sized {
        let () = x;
    }
}

mod ex2 {
    type Tait1 = impl Sized;
    //[current]~^ ERROR unconstrained opaque type
    type Tait2 = impl Sized;
    #[define_opaque(Tait1, Tait2)]
    fn foo(x: Tait1) -> Tait2 {
        let () = x;
    }
}

mod ex3 {
    type Tait1 = impl Sized;
    //[current]~^ ERROR unconstrained opaque type
    trait Something<T> {}
    impl<T, U> Something<U> for T {}
    type Tait2 = impl Something<Tait1>;
    #[define_opaque(Tait1, Tait2)]
    fn foo(x: Tait1) -> Tait2 {
        let () = x;
    }
}

mod ex4 {
    type Tait1 = impl Sized;
    //[current]~^ ERROR unconstrained opaque type
    trait Trait<U> {
        type Assoc;
    }

    impl<T, U> Trait<U> for T {
        type Assoc = T;
    }

    // ambiguity when checking that `Tait2` is wf
    //
    // ambiguity proving `(): Trait<Tait1>`.
    type Tait2 = impl Trait<(), Assoc = impl Trait<Tait1>>;
    #[define_opaque(Tait1, Tait2)]
    fn foo(x: Tait1) -> Tait2 {
        let () = x;
    }
}

fn main() {}
