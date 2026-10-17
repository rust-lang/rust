// Exercise implicit trait object lifetime bounds inside type-relative assoc ty paths.

mod own { // the lifetime comes from the own generics
    trait Outer { type Ty<'a, T: 'a + ?Sized>; }
    trait Inner {}

    fn f<'r, T: Outer>(x: T::Ty<'r, dyn Inner + 'r>) { /*check*/ g::<T>(x) }
    // FIXME: Ideally, we would deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty
    //        `Ty` but for that we'd need to somehow obtain the resolution of the type-relative path
    //        `T::Ty` from HIR ty lowering in RBV (it resolves to `<T as Outer>::Ty`).
    fn g<'r, T: Outer>(x: T::Ty<'r, dyn Inner>) {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context
}

mod parent { // the lifetime comes from the parent generics
    trait Outer<'a> { type Ty<T: 'a + ?Sized>; }
    trait Inner {}

    fn f<'r, T: Outer<'r>>(x: T::Ty<dyn Inner + 'r>) { /*check*/ g::<T>(x) }
    // FIXME: Ideally, we would deduce `dyn Inner + 'r` from the bound `'a` on ty param `T` of assoc
    //        ty `Ty` but for that we'd need to somehow obtain the resolution of the type-relative
    //        path `T::Ty` from HIR ty lowering in RBV (it resolved to `<T as Outer<'r>>::Ty`).
    fn g<'r, T: Outer<'r>>(_: T::Ty<dyn Inner>) {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context

    // Assuming RBV was able to resolve type-relative paths, it should not crash on "overly lifetime
    // polymorphic" paths (`for<'r> <T as Outer<'r>>::Ty<dyn Inner + 'r>` isn't a valid Rust type).
    fn escaping<T: for<'r> Outer<'r>>(_: T::Ty<dyn Inner>) {}
    //~^ ERROR cannot use the associated type of a trait with uninferred generic parameters
}

fn main() {}
