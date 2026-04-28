//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod test_1 {
    trait Trait<'b, T>: Sized {
        fn foo<U, const M: bool>(&self) {}
    }

    impl<'a, T> Trait<'a, T> for u8 {}

    reuse Trait::foo::<String, false> as bar1;
    reuse Trait::foo as bar2;
    reuse Trait::<'static, ()>::foo::<String, false> as bar3;
    reuse Trait::<'static, ()>::foo as bar4;

    fn check1() {
        // Method call is generated but parent generics too as Self is present.
        bar1::<'static, u8, ()>(&1);
        bar2::<'static, u8, (), (), false>(&2);
        // Method call is not generated as generic args specified in parent.
        bar3::<u8>(&3);
        bar4::<u8, (), false>(&4);
    }

    reuse <u8 as Trait>::foo::<String, false> as bar5;
    reuse <u8 as Trait>::foo as bar6;
    reuse <u8 as Trait::<'static, ()>>::foo::<String, false> as bar7;
    reuse <u8 as Trait::<'static, ()>>::foo as bar8;

    fn check2() {
        // Method call is not generated as qself present.
        bar5::<'static, ()>(&1);
        bar6::<'static, (), (), false>(&2);
        bar7(&3);
        bar8::<(), false>(&4);
    }

    trait Trait2<'a, 'b, 'c, X, Y, Z>: Sized {
        fn get() -> &'static u8 {
            &0
        }
        fn get_self(&self) -> &'static u8 {
            &0
        }

        reuse Trait::foo::<String, false> as bar1 { Self::get() }
        //~^ ERROR: type annotations needed
        reuse Trait::foo as bar2 { Self::get() }
        //~^ ERROR: type annotations needed
        reuse Trait::<'static, ()>::foo::<String, false> as bar3 { Self::get() }
        reuse Trait::<'static, ()>::foo as bar4 { Self::get() }

        reuse Trait::foo::<String, false> as bar5 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::foo as bar6 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::<'static, ()>::foo::<String, false> as bar7 { self.get_self() }
        reuse Trait::<'static, ()>::foo as bar8 { self.get_self() }

        reuse <u8 as Trait>::foo::<String, false> as bar9 { Self::get() }
        reuse <u8 as Trait>::foo as bar10 { Self::get() }
        reuse <u8 as Trait::<'static, ()>>::foo::<String, false> as bar11 { Self::get() }
        reuse <u8 as Trait::<'static, ()>>::foo as bar12 { Self::get() }

        reuse <u8 as Trait>::foo::<String, false> as bar13 { self.get_self() }
        reuse <u8 as Trait>::foo as bar14 { self.get_self() }
        reuse <u8 as Trait::<'static, ()>>::foo::<String, false> as bar15 { self.get_self() }
        reuse <u8 as Trait::<'static, ()>>::foo as bar16 { self.get_self() }

        fn check(&self) {
            // Method call is generated so no parent generics.
            self.bar1();
            self.bar2::<(), false>();
            // Method call is not generated as generic args specified in parent.
            self.bar3();
            self.bar4::<(), false>();

            // Method call is generated so no parent generics.
            self.bar5();
            self.bar6::<(), false>();
            // Method call is not generated as generic args specified in parent.
            self.bar7();
            self.bar8::<(), false>();

            // Method call is not generated (because qself is present).
            self.bar9::<'static, ()>();
            self.bar10::<'static, (), (), false>();
            self.bar11();
            self.bar12::<(), false>();

            // Method call is not generated (because qself is present).
            self.bar13::<'static, ()>();
            self.bar14::<'static, (), (), false>();
            self.bar15();
            self.bar16::<(), false>();
        }
    }

    struct X;

    impl X {
        fn get() -> &'static u8 {
            &0
        }
        fn get_self(&self) -> &'static u8 {
            &0
        }

        reuse Trait::foo::<String, false> as bar1 { Self::get() }
        //~^ ERROR: type annotations needed
        reuse Trait::foo as bar2 { Self::get() }
        //~^ ERROR: type annotations needed
        reuse Trait::<'static, usize>::foo::<String, false> as bar3 { Self::get() }
        reuse Trait::<'static, usize>::foo as bar4 { Self::get() }

        reuse Trait::foo::<String, false> as bar5 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::foo as bar6 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::<'static, usize>::foo::<String, false> as bar7 { self.get_self() }
        reuse Trait::<'static, usize>::foo as bar8 { self.get_self() }

        reuse <u8 as Trait>::foo::<String, false> as bar9 { Self::get() }
        reuse <u8 as Trait>::foo as bar10 { Self::get() }
        reuse <u8 as Trait::<'static, usize>>::foo::<String, false> as bar11 { Self::get() }
        reuse <u8 as Trait::<'static, usize>>::foo as bar12 { Self::get() }

        reuse <u8 as Trait>::foo::<String, false> as bar13 { self.get_self() }
        reuse <u8 as Trait>::foo as bar14 { self.get_self() }
        reuse <u8 as Trait::<'static, usize>>::foo::<String, false> as bar15 { self.get_self() }
        reuse <u8 as Trait::<'static, usize>>::foo as bar16 { self.get_self() }

        fn check(&self) {
            // Method call is generated so no parent generics.
            self.bar1();
            self.bar2::<(), false>();
            // Method call is not generated as generic args specified in parent.
            self.bar3();
            self.bar4::<(), false>();

            // Method call is generated so no parent generics.
            self.bar5();
            self.bar6::<(), false>();
            // Method call is not generated as generic args specified in parent.
            self.bar7();
            self.bar8::<(), false>();

            // Method call is not generated (because qself is present).
            self.bar9::<'static, ()>();
            self.bar10::<'static, (), (), false>();
            self.bar11();
            self.bar12::<(), false>();

            // Method call is not generated (because qself is present).
            self.bar13::<'static, ()>();
            self.bar14::<'static, (), (), false>();
            self.bar15();
            self.bar16::<(), false>();
        }
    }
}

mod test_2 {
    trait Trait<'b, T>: Sized {
        fn foo<U, const M: bool>(&self) {}
    }

    impl Trait<'static, usize> for u8 {}
    impl Trait<'static, String> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z>: Sized {
        fn get_self(&self) -> &'static u8 {
            &0
        }

        reuse Trait::foo::<String, false> as bar1 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::foo as bar2 { self.get_self() }
        //~^ ERROR: type annotations needed
        reuse Trait::<'static, usize>::foo::<String, false> as bar3 { self.get_self() }
        reuse Trait::<'static, ()>::foo as bar4 { self.get_self() }
        //~^ ERROR: the trait bound `u8: test_2::Trait<'static, ()>` is not satisfied

        fn check(&self) {
            // Method call is generated so no parent generics.
            self.bar1();
            self.bar2::<(), false>();
            // Method call is not generated as generic args specified in parent.
            self.bar3();
            self.bar4::<(), false>();
        }
    }
}

// Testing that if other unused in signature generic param
// is in bounds of other used parent's generic param it will be generated.
mod test_3 {
    trait Bound<T> {}
    trait Trait<A: Bound<C>, B: Bound<A>, C> {
        fn foo<U, const M: bool>(&self, b: B) {}
    }

    impl<A: Bound<C>, B: Bound<A>, C> Trait<A, B, C> for usize {}

    impl<T> Bound<T> for usize {}
    impl<T> Bound<T> for () {}

    trait Trait2<'a, 'b, 'c, X, Y, Z>: Sized {
        fn get_self(&self) -> &'static u8 {
            &0
        }

        fn get_self2(&self) -> &'static usize {
            &0
        }

        reuse Trait::foo::<String, false> as bar1 { self.get_self() }
        //~^ ERROR: no method named `foo` found for reference `&'static u8` in the current scope
        reuse Trait::foo as bar2 { self.get_self() }
        //~^ ERROR: no method named `foo` found for reference `&'static u8` in the current scope
        reuse Trait::<(), usize, ()>::foo::<String, false> as bar3 { self.get_self() }
        //~^ ERROR: the trait bound `u8: test_3::Trait<(), usize, ()>` is not satisfied
        reuse Trait::<(), (), ()>::foo as bar4 { self.get_self() }
        //~^ ERROR: the trait bound `u8: test_3::Trait<(), (), ()>` is not satisfied

        reuse Trait::<(), usize, ()>::foo as bar5 { self.get_self() }
        //~^ ERROR: the trait bound `u8: test_3::Trait<(), usize, ()>` is not satisfied

        reuse Trait::foo::<String, false> as bar6 { self.get_self2() }
        reuse Trait::foo as bar7 { self.get_self2() }
        reuse Trait::<(), usize, ()>::foo::<String, false> as bar8 { self.get_self2() }
        reuse Trait::<(), (), ()>::foo as bar9 { self.get_self2() }
        reuse Trait::<(), usize, ()>::foo as bar10 { self.get_self2() }

        fn check(&self) {
            // Method call is generated and needed parent generics are generated.
            self.bar1::<(), (), ()>(());
            self.bar2::<(), (), (), (), false>(());
            // Method call is not generated as generic args specified in parent.
            self.bar3(123);
            self.bar4::<(), false>(());
            self.bar5::<(), false>(123);

            // Method call is generated and needed parent generics are generated.
            self.bar6::<(), (), ()>(());
            self.bar7::<(), (), (), (), false>(());
            // Method call is not generated as generic args specified in parent.
            self.bar8(123);
            self.bar9::<(), false>(());
        }
    }
}

// Testing that lifetimes from predicates are generated.
mod test_4 {
    trait Bound<'a> {}
    trait Trait<'a, 'b, C: 'a + 'static, B: Bound<'b>> {
        fn foo<U, const M: bool>(&self, b: B, c: C) {}
    }

    impl<'a, 'b, C: 'a + 'static, B: Bound<'b>> Trait<'a, 'b, C, B> for () {}
    impl Bound<'static> for () {}

    struct X;
    impl X {
        fn get_self() -> () {
            ()
        }

        reuse Trait::foo { Self::get_self() }

        fn check(&self) {
            self.foo::<'static, 'static, (), (), (), false>((), ());
        }
    }
}

// Testing that params from region outlives predicates are generated.
mod test_5 {
    trait Trait<'a, 'b: 'static + 'a> {
        fn foo<U, const M: bool>(&self, b: &'b str) {}
    }

    impl<'a, 'b: 'a + 'static> Trait<'a, 'b> for () {}

    struct X;
    impl X {
        fn get_self() -> () {
            ()
        }

        reuse Trait::foo { Self::get_self() }

        fn check(&self) {
            self.foo::<'static, 'static, (), false>("");
        }
    }
}

// Testing with impl traits.
mod test_6 {
    trait Bound<B> {}
    trait Trait<'a, 'b: 'static + 'a, A: Bound<B>, B, C> {
        fn foo<U, const M: bool>(&self, b: &'b str, f: impl FnOnce() -> usize) {}
        fn bar<const M: bool>(&self, f: impl FnOnce() -> A) {}
    }

    impl<'a, 'b: 'a + 'static, A: Bound<B>, B, C> Trait<'a, 'b, A, B, C> for () {}

    impl Bound<u8> for usize {}

    struct X;
    impl X {
        fn get_self() -> () {
            ()
        }

        // `C` is not generated so type annotation needed error.
        reuse Trait::* { Self::get_self() }
        //~^ ERROR: type annotations needed
        //~| ERROR: type annotations needed

        fn check(&self) {
            self.foo::<'static, 'static, (), false>("", || 123);

            self.bar::<usize, u8, true>(|| 123);
            self.bar::<usize, u32, true>(|| 123);
            //~^ ERROR: the trait bound `usize: test_6::Bound<u32>` is not satisfied
        }
    }

    impl Trait<'static, 'static, usize, u8, String> for ((), ()) {}
    struct Y;
    impl Y {
        fn get_self() -> ((), ()) {
            ((), ())
        }

        // `C` is not generated, but not type annotation error as there is a single
        // implementation of `Trait` for `((), ())`, however as `Trait` generic args
        // are not specified we generate generic param `A` and thus the function type
        // is still `FnOnce() -> A` (not `FnOnce() -> usize`), so we get an error.
        reuse Trait::* { Self::get_self() }
        //~^ ERROR: expected `impl FnOnce() -> A` to return `usize`, but it returns `A`

        fn check(&self) {
            self.foo::<'static, 'static, (), false>("", || 123);

            self.bar::<usize, u8, true>(|| 123);
            self.bar::<usize, u32, true>(|| 123);
            //~^ ERROR: the trait bound `usize: test_6::Bound<u32>` is not satisfied
        }

        // Method call is not generated as `Trait`'s args are specified, so we have to explicitly
        // use `&Self::get_self()` instead of `Self::get_self()`.

        // FIXME(fn_delegation): do we still want this behavior? We need those args for signature
        // inheritance as we use HirId of path segment to extract them, but we can probably
        // do it at delayed AST -> HIR lowering, moreover we would need those args for
        // inherent methods resolution, so we will have to obtain them at AST -> HIR lowering
        // anyway.
        reuse Trait::<'static, 'static, usize, u8, String>::bar as bar1 { &Self::get_self() }

        fn check1(&self) {
            self.bar1::<false>(|| 123usize);
        }
    }
}

fn main() {}
