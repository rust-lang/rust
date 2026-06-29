//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]

// Some interesting cases:
mod selected_tests {
    mod different_infers {
        fn foo<'a, 'b: 'b, 'c, X, const M: usize, Y>(_: &'a &'b &'c ()) {}

        // Should differentiate between lifetime and types/consts infers.
        reuse foo::<_, '_, '_, '_> as bar;
        //~^ ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
    }

    mod self_type {
        trait Trait<'a, X> {
            fn method<'b: 'b, const M: usize>(&self) {}
            fn r#static<'b, Y, const B: bool>() {}
        }

        impl<'a, X> Trait<'a, X> for () {}

        reuse Trait::<'_, _>::method::<'_, _> as foo;

        reuse <_ as Trait<'_, _>>::method::<'_, _> as foo1;
        reuse <() as Trait<'_, _>>::method::<'_, _> as foo2;

        reuse <_ as Trait<'_, _>>::r#static::<_, _> as foo3;
        reuse <() as Trait<'_, _>>::r#static::<_, _> as foo4;

        reuse Trait::<'_, _>::r#static::<_, _> as foo5;
    }

    mod late_bound_lifetimes {
        fn foo<'a, 'b, 'c: 'c, 'd>(_: &'a &'b &'c &'d ()) {}

        // 'c corresponds to infer.
        reuse foo::<'_> as foo1;

        // Only 'c is generated in desugaring, second infer remains just infer in call path.
        reuse foo::<'_, '_> as foo2;
        //~^ ERROR: function takes 1 lifetime argument but 2 lifetime arguments were supplied

        reuse foo as foo3;
        reuse foo::<'static> as foo4;
    }

    mod non_angle_bracketed_args {
        fn foo<'a, 'b: 'b, 'c, X, const M: usize, Y>(_: &'a &'b &'c ()) {}

        reuse foo::('_, _, _, _) as bar;
        //~^ ERROR: lifetimes must be followed by `+` to form a trait object type
        //~| ERROR: at least one trait is required for an object type
        //~| ERROR: parenthesized type parameters may only be used with a `Fn` trait [E0214]
        //~| ERROR: function takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
        //~| ERROR: function takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
    }
}

// All other stuff:
mod legacy_tests {
    trait Trait<T> {
        fn foo<U>(&self, _: U, _: T) {}
    }

    impl<T> Trait<T> for u8 {}

    reuse Trait::<_>::foo::<i32> as generic_arguments1;
    reuse <u8 as Trait<_>>::foo as generic_arguments2;
    reuse <_ as Trait<_>>::foo as generic_arguments3;
}

mod free_to_free {
    fn foo<'a, 'b: 'b, 'c, X, const M: usize, Y>(_: &'a &'b &'c ()) {}

    reuse foo::<> as foo1;
    reuse foo::<'_, _, _, _> as foo2;
    reuse foo::<'static, String, _, _> as foo3;
    reuse foo::<'_, _, 123, _> as foo4;

    reuse foo::<'_, '_, '_, _, _, _,> as foo5;
    //~^ ERROR: function takes 3 generic arguments but 5 generic arguments were supplied
    //~| ERROR: wrong infer used: expected _, found: '_
    //~| ERROR: wrong infer used: expected _, found: '_

    reuse foo::<_, _, _, '_, '_, '_, _, _, _,> as foo6;
    //~^ ERROR: function takes 3 generic arguments but 6 generic arguments were supplied [E0107]
    //~| ERROR: function takes 1 lifetime argument but 3 lifetime arguments were supplied
    //~| ERROR: wrong infer used: expected '_, found: _
    //~| ERROR: wrong infer used: expected _, found: '_

    reuse foo::<_, '_, _, _> as foo7;
    //~^ ERROR: wrong infer used: expected '_, found: _
    //~| ERROR: wrong infer used: expected _, found: '_

    reuse foo::<'_, '_, '_, '_> as foo8;
    //~^ ERROR: wrong infer used: expected _, found: '_
    //~| ERROR: wrong infer used: expected _, found: '_
    //~| ERROR: wrong infer used: expected _, found: '_

    reuse foo::<_> as foo9;
    //~^ ERROR: function takes 3 generic arguments but 0 generic arguments were supplied
    //~| ERROR: wrong infer used: expected '_, found: _

    reuse foo::<Vec<'_>, _, _, ()> as foo10;
    //~^ ERROR: function takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
    //~| ERROR: function takes 3 generic arguments but 4 generic arguments were supplied [E0107]
    //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    //~| ERROR: struct takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    //~| ERROR: struct takes at least 1 generic argument but 0 generic arguments were supplied [E0107]

    reuse foo::<Vec<_>, _, _, ()> as foo11;
    //~^ ERROR: function takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
    //~| ERROR: function takes 3 generic arguments but 4 generic arguments were supplied [E0107]
    //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]

    reuse foo::<'____, ___, _, ___> as foo12;
    //~^ ERROR: use of undeclared lifetime name `'____` [E0261]
    //~| ERROR: cannot find type `___` in this scope [E0425]
    //~| ERROR: cannot find type `___` in this scope [E0425]

    reuse foo::<'_, Vec<_>, Vec<Vec<_>>, _> as foo13;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
    //~| ERROR: type provided when a constant was expected [E0747]

    reuse foo::<'_, unresolved_, _, _> as foo14;
    //~^ ERROR: cannot find type `unresolved_` in this scope

    reuse foo::<_, _, _> as foo15;
    //~^ ERROR: function takes 3 generic arguments but 2 generic arguments were supplied
    //~| ERROR: wrong infer used: expected '_, found: _
}

mod free_to_trait {
    pub trait Trait<'a, 'b, X, const C: usize, Y> {
        fn foo<'aa, 'bb: 'bb, 'cc, XX, const M: usize, YY>(&self, _: &'aa &'b &'cc ()) {}
    }

    struct X;
    impl<'a, 'b, Some, Params, X, const C: usize, Y> Trait<'a, 'b, X, C, Y> for X {}
    //~^ ERROR: the type parameter `Some` is not constrained by the impl trait, self type, or predicates [E0207]
    //~| ERROR: the type parameter `Params` is not constrained by the impl trait, self type, or predicates [E0207]

    mod child_only {
        use super::*;

        reuse Trait::foo::<> as foo1;
        reuse Trait::foo::<'_, _, _, _> as foo2;
        reuse Trait::foo::<'static, String, _, _> as foo3;
        reuse Trait::foo::<'_, _, 123, _> as foo4;

        reuse Trait::foo::<'_, '_, '_, _, _, _,> as foo5;
        //~^ ERROR: method takes 3 generic arguments but 5 generic arguments were supplied
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::foo::<_, _, _, '_, '_, '_, _, _, _,> as foo6;
        //~^ ERROR: method takes 3 generic arguments but 6 generic arguments were supplied [E0107]
        //~| ERROR: method takes 1 lifetime argument but 3 lifetime arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::foo::<_, '_, _, _> as foo7;
        //~^ ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::foo::<'_, '_, '_, '_> as foo8;
        //~^ ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::foo::<_> as foo9;
        //~^ ERROR: method takes 3 generic arguments but 0 generic arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _

        reuse Trait::foo::<Vec<'_>, _, _, ()> as foo10;
        //~^ ERROR: method takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: struct takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
        //~| ERROR: struct takes at least 1 generic argument but 0 generic arguments were supplied [E0107]
        //~| ERROR: method takes 1 lifetime argument but 0 lifetime arguments were supplied

        reuse Trait::foo::<Vec<_>, _, _, ()> as foo11;
        //~^ ERROR: method takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
        //~| ERROR: method takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]

        reuse Trait::foo::<'____, ___, _, ___> as foo12;
        //~^ ERROR: use of undeclared lifetime name `'____` [E0261]
        //~| ERROR: cannot find type `___` in this scope [E0425]
        //~| ERROR: cannot find type `___` in this scope [E0425]

        reuse Trait::foo::<'_, Vec<_>, Vec<Vec<_>>, _> as foo13;
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: type provided when a constant was expected [E0747]

        reuse Trait::foo::<'_, unresolved_, _, _> as foo14;
        //~^ ERROR: cannot find type `unresolved_` in this scope

        reuse Trait::foo::<_, _, _> as foo15;
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _
    }

    mod parent_only {
        use super::*;

        reuse Trait::<'_, 'static, _, _, _>::foo as foo1;
        reuse Trait::<'_, '_, _, _, _>::foo as foo2;

        reuse Trait::<'_, (), _, '_, _>::foo as foo3;
        //~^ ERROR: trait takes 2 lifetime arguments but 1 lifetime argument was supplied [E0107]
        //~| ERROR: trait takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<>::foo as foo4;

        reuse Trait::<_, _>::foo as foo5;
        //~^ ERROR: trait takes 3 generic arguments but 0 generic arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected '_, found: _

        reuse Trait::<'_, '_>::foo as foo6;
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions

        reuse Trait::<'_, '_, Vec<_>, 123, Vec<Vec<_>>>::foo as foo7;
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]

        reuse Trait::<'static, 'static, (), 123, ()>::foo as foo8;
        reuse Trait::<'static, 'static, _, _, _>::foo as foo9;

        reuse Trait::<'static, 'static, _, _, _, _, _, _, _>::foo as foo10;
        //~^ ERROR: trait takes 3 generic arguments but 7 generic arguments were supplied

        reuse Trait::<'static, 'static, '_,'_, '_, '_, '_, '_, '_>::foo as foo11;
        //~^ ERROR: trait takes 2 lifetime arguments but 6 lifetime arguments were supplied
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'static, 'static, _>::foo as foo12;
        //~^ ERROR: trait takes 3 generic arguments but 1 generic argument was supplied
    }

    mod parent_and_child_random {
        use super::*;

        reuse Trait::<'_, 'static, _, _, _>::foo::<> as foo1;
        reuse Trait::<'_, '_, _, _, _>::foo::<'_, _, _, _> as foo2;

        reuse Trait::<'_, (), _, '_, _>::foo::<'static, String, _, _> as foo3;
        //~^ ERROR: trait takes 2 lifetime arguments but 1 lifetime argument was supplied [E0107]
        //~| ERROR: trait takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<>::foo::<'_, _, 123, _> as foo4;

        reuse Trait::<_, _>::foo::<'_, '_, '_, _, _, _,> as foo5;
        //~^ ERROR: trait takes 3 generic arguments but 0 generic arguments were supplied
        //~| ERROR: method takes 3 generic arguments but 5 generic arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'_, '_>::foo::<_, _, _, '_, '_, '_, _, _, _,> as foo6;
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: method takes 3 generic arguments but 6 generic arguments were supplied [E0107]
        //~| ERROR: method takes 1 lifetime argument but 3 lifetime arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'_, '_, Vec<_>, 123, Vec<Vec<_>>>::foo::<_, '_, _, _> as foo7;
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: wrong infer used: expected '_, found: _
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'static, 'static, (), 123, ()>::foo::<'_, '_, '_, '_> as foo8;
        //~^ ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'static, 'static, _, _, _>::foo::<_> as foo9;
        //~^ ERROR: method takes 3 generic arguments but 0 generic arguments were supplied
        //~| ERROR: wrong infer used: expected '_, found: _

        reuse Trait::<'static, 'static, _, _, _, _, _, _, _>::foo::<Vec<'_>, _, _, ()> as foo10;
        //~^ ERROR: trait takes 3 generic arguments but 7 generic arguments were supplied [E0107]
        //~| ERROR: method takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
        //~| ERROR: method takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: struct takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
        //~| ERROR: struct takes at least 1 generic argument but 0 generic arguments were supplied [E0107]

        reuse Trait::<'static, 'static, '_,'_, '_, '_, '_, '_>::foo::<Vec<_>, _, _, ()> as foo11;
        //~^ ERROR: trait takes 2 lifetime arguments but 5 lifetime arguments were supplied [E0107]
        //~| ERROR: method takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]
        //~| ERROR: method takes 3 generic arguments but 4 generic arguments were supplied [E0107]
        //~| ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for functions [E0121]
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_
        //~| ERROR: wrong infer used: expected _, found: '_

        reuse Trait::<'static, 'static, _>::foo::<'____, ___, _, ___> as foo12;
        //~^ ERROR: cannot find type `___` in this scope
        //~| ERROR: cannot find type `___` in this scope
        //~| ERROR: use of undeclared lifetime name `'____`
        //~| ERROR: trait takes 3 generic arguments but 1 generic argument was supplied
    }
}

mod trait_impl_to_free {
    pub trait Trait<'a, 'b, X, const C: usize, Y> {
        fn foo<'aa, 'bb: 'bb, 'cc, XX, const M: usize, YY>(&self) {}
    }

    struct S;
    impl<'a, 'b, X, const C: usize, Y> Trait<'a, 'b, X, C, Y> for S {}

    mod to_reuse {
        pub fn foo<X, const M: usize, Y>(_: ()) {}
    }

    struct F1(S);
    impl<'a, 'b, X, const C: usize, Y> Trait<'a, 'b, X, C, Y> for F1 {
        reuse to_reuse::foo::<_, _, _> { self.0 }
        //~^ ERROR: mismatched types
    }

    struct F2(S);
    impl<'a, 'b, X, const C: usize, Y> Trait<'a, 'b, X, C, Y> for F2 {
        reuse to_reuse::foo { self.0 }
        //~^ ERROR: mismatched types
        //~| ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied
    }

    struct F3(S);
    impl<'a, 'b, X, const C: usize, Y> Trait<'a, 'b, X, C, Y> for F3 {
        reuse to_reuse::foo::<(), 123, ()> { self.0 }
        //~^ ERROR: mismatched types
    }
}

fn main() {}
