#![feature(late_bound_turbofishing)]

fn require_static<T: 'static>(_: T) { }
fn foo_early<'a: 'a>(b: &'a u32) -> &'a u32 { b }
fn foo_late<'a>(b: &'a u32) -> &'a u32 { b }
fn foo_latest(_: &u32) {}

mod ex1 {
    // compiles without errors

    trait Trait {
        type Assoc<'a>;
    }

    // zero explicit generic lifetimes
    fn do_thing<T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> &u32 {
        todo!()
    }

    fn foo<T: Trait>() {
        // one explicit generic lifetime
        do_thing::<'static, T>(None);
        //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    }
}

mod ex2 {
    // compiles without errors

    trait Trait {
        type Assoc<'a>;
    }

    // one explicit generic lifetime
    fn do_thing<'b, T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> (&u32, &'b i64) {
        todo!()
    }

    fn foo<T: Trait>() {
        // two explicit generic lifetimes
        do_thing::<'static, 'static, T>(None);
        //~^ ERROR: function takes 1 lifetime argument but 2 lifetime arguments were supplied [E0107]
    }
}

mod ex3 {
    // compiles without errors

    trait Trait {
        type Assoc<'a>;
    }

    // zero explicit generic lifetimes
    fn do_thing<T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> u32 {
        todo!()
    }

    fn foo<T: Trait>() {
        // one explicit generic lifetime
        do_thing::<'static, T>(None);
        //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied
    }
}

mod ex4 {
    // compiles without errors

    trait Trait {
        type Assoc<'a>;
        // zero explicit generic lifetimes
        fn do_thing(_: Option<Self::Assoc<'_>>) -> &u32 {
            todo!()
        }
    }

    fn foo<T: Trait>() {
        // one explicit generic lifetime
        <T as Trait>::do_thing::<'static>(None);
        //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied
    }
}

mod ex5 {
    // compiles without errors

    trait Trait {
        type Assoc;
    }

    // zero explicit generic lifetimes
    fn do_thing<T>(_: Option<<&T as Trait>::Assoc>) -> &u32
    where
        for<'c> &'c T: Trait,
    {
        todo!()
    }

    fn foo<T: 'static>()
    where
        for<'c> &'c T: Trait,
    {
        // one explicit generic lifetime
        do_thing::<'static, T>(None);
        //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied
    }
}

mod ex6 {
    // compiles without errors

    trait Trait {
        type Assoc<'a>;
        // FIXME
        // zero explicit generic lifetimes
        fn do_thing(_: Option<Self::Assoc<'_>>) -> &u32;
    }

    impl Trait for u32 {
        type Assoc<'a> = i64;
        // one explicit generic lifetime
        fn do_thing<'b>(_: Option<i64>) -> &'b u32 {
            todo!()
        }
    }
}

mod ex7 {
    pub struct ReqLtInvariant<'a, T>(&'a mut (*mut T));

    fn require_static<T: 'static>(_: T) {}

    fn require_exact<'a, T: 'a>(_: T) -> ReqLtInvariant<'a, T> {
        ReqLtInvariant(todo!())
    }

    fn foo<'a>(b: &'a u32) -> &'a u32 { b }

    fn bar<'a>(n: &'a u32) {
        let f1 = foo::<'static>;
        require_static(f1);
        require_exact::<'a>(f1);
        // ^ this should not compile
        f1(n);

        let f2 = foo::<'a>;
        require_exact::<'a>(f2);
        f2(n);
    }

    fn munch() {
        let f = foo::<'static>;
        let freerf = 4u32;
        bar(&freerf);
    }
}

fn bar<'a>(_: &'a u32) {
    let f = foo_late::<'a>;
    require_static(f);
    // ^ FIXME: This SHOULD NOT COMPILE because it is UNSOUND but it does anyway.
    //   this is related to how FnDef has broken outlives checking.
}

fn main() {
    let f = foo_early::<'static>;
    require_static(f);
    let f = foo_late::<'static>;
    require_static(f);
    let f = foo_latest::<'static>;
    //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    require_static(f);
    {
        bar(&4)
    }
}
