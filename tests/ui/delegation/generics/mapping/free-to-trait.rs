#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing lifetimes + types + consts in both parent and child, reusing in
// a function without generic params
mod test_1 {
    trait Trait<'b, 'c, 'a, T, const N: usize>: Sized {
        fn foo<'d: 'd, U, const M: bool>(self) {}
    }

    impl Trait<'static, 'static, 'static, i32, 1> for u8 {}

    pub fn check() {
        fn no_ctx() {
            reuse Trait::foo as bar;
            //~^ ERROR: type annotations needed [E0284]
            bar::<'static, 'static, 'static, 'static, u8, i32, 1, String, true>(123);
        }

        fn with_ctx<'a, 'b, 'c, A, B, C, const N: usize, const M: bool>() {
            reuse Trait::foo as bar;
            //~^ ERROR: type annotations needed [E0284]
            bar::<'static, 'static, 'static, 'a, u8, i32, 1, A, M>(123);
        }

        no_ctx();
        with_ctx::<i32, i32, i32, 1, true>();
    }
}

// Testing lifetimes + types + consts in both parent and child, add user-specified args to parent
mod test_2 {
    trait Trait<'a, T, const N: usize>: Sized {
        fn foo<'b: 'b, U, const M: bool>(self) {}
    }

    impl Trait<'static, i32, 1> for u8 {}

    pub fn check() {
        reuse Trait::<'static, i32, 1>::foo as bar;
        //~^ ERROR: the trait bound `Self: test_2::Trait<'static, i32, 1>` is not satisfied [E0277]

        bar::<'static, u8, String, true>(123);
        //~^ ERROR: function takes 5 generic arguments but 3 generic arguments were supplied [E0107]
        //~| ERROR: function takes 2 lifetime arguments but 1 lifetime argument was supplied [E0107]
    }
}

// Testing lifetimes + types + consts in both parent and child,
// add user-specified args to child
mod test_3 {
    trait Trait<'a, T, const N: usize>: Sized {
        fn foo<'b: 'b, U, const M: bool>(self) {}
    }

    impl Trait<'static, String, 1> for u8 {}

    pub fn check() {
        reuse Trait::foo::<'static, i32, true> as bar;

        bar::<'static, u8, String, 1>(123);
        //~^ ERROR: function takes 5 generic arguments but 3 generic arguments were supplied [E0107]
        //~| ERROR: function takes 2 lifetime arguments but 1 lifetime argument was supplied [E0107]
    }
}

// Testing types/consts in parent, lifetimes + types/consts in child,
// add user-specified args to child
mod test_4 {
    trait Trait<T, const N: usize>: Sized {
        fn foo<'b: 'b, U, const M: bool>(self) {}
    }

    impl Trait<String, 1> for u8 {}

    pub fn check() {
        reuse Trait::foo::<'static, i32, true> as bar;

        bar::<u8, String, 1>(123);
        //~^ ERROR: function takes 5 generic arguments but 3 generic arguments were supplied [E0107]
    }
}

// Testing consts in parent, lifetimes + types/consts in child, add user-specified args to child
mod test_5 {
    trait Trait<const N: usize>: Sized {
        fn foo<'b: 'b, U, const M: bool>(self) {}
    }

    impl Trait<1> for u8 {}

    pub fn check() {
        reuse Trait::foo::<'static, i32, true> as bar;

        bar::<u8, 1>(123);
        //~^ ERROR: function takes 4 generic arguments but 2 generic arguments were supplied [E0107]
    }
}

// Testing no generics in parent, lifetimes + types/consts in child,
// add user-specified args to child
mod test_6 {
    trait Trait: Sized {
        fn foo<'b: 'b, U, const M: bool>(self) {}
    }

    impl Trait for u8 {}

    pub fn check() {
        reuse Trait::foo::<'static, i32, true> as bar;

        bar::<u8>(123);
        //~^ ERROR: function takes 3 generic arguments but 1 generic argument was supplied [E0107]
    }
}

// Testing lifetimes + types/consts in parent, types/consts in child,
// add user-specified args to parent
mod test_7 {
    trait Trait<'a, T, const N: usize>: Sized {
        fn foo<U, const M: bool>(self) {}
    }

    impl Trait<'static, i32, 1> for u8 {}

    pub fn check() {
        reuse Trait::<'static, i32, 1>::foo as bar;
        //~^ ERROR: the trait bound `Self: test_7::Trait<'static, i32, 1>` is not satisfied [E0277]

        bar::<u8, String, true>(123);
        //~^ ERROR: function takes 5 generic arguments but 3 generic arguments were supplied [E0107]
    }
}

// Testing lifetimes + types/consts in parent, consts in child, add user-specified args to parent
mod test_8 {
    trait Trait<'a, T, const N: usize>: Sized {
        fn foo<const M: bool>(self) {}
    }

    impl Trait<'static, i32, 1> for u8 {}

    pub fn check() {
        reuse Trait::<'static, i32, 1>::foo as bar;
        //~^ ERROR: the trait bound `Self: test_8::Trait<'static, i32, 1>` is not satisfied [E0277]

        bar::<u8, true>(123);
        //~^ ERROR: function takes 4 generic arguments but 2 generic arguments were supplied [E0107]
    }
}

// Testing lifetimes + types/consts in parent, none in child, add user-specified args to parent
mod test_9 {
    trait Trait<'a, T, const N: usize>: Sized {
        fn foo(self) {}
    }

    impl Trait<'static, i32, 1> for u8 {}

    pub fn check() {
        reuse Trait::<'static, i32, 1>::foo as bar;
        //~^ ERROR: the trait bound `Self: test_9::Trait<'static, i32, 1>` is not satisfied [E0277]

        bar::<u8>(123);
        //~^ ERROR: function takes 3 generic arguments but 1 generic argument was supplied [E0107]
    }
}

// Testing lifetimes + types in parent, lifetimes + types/consts in child,
// adding self ty to reuse, testing using generic params from delegation parent
// context, adding user-specified args to none, parent, parent and child
mod test_10 {
    trait Trait<'b, 'c, T> {
        fn foo<'d: 'd, U, const M: bool>() {}
    }

    impl<'b, 'c, T> Trait<'b, 'c, T> for u8 {}

    pub fn check() {
        fn with_ctx<'a, 'b, 'c, A, B, C, const N: usize, const M: bool>() {
            reuse <u8 as Trait>::foo as bar;
            //~^ ERROR: missing generics for trait `test_10::Trait` [E0107]
            bar::<'a, 'b, 'c, u8, C, A, M>();
            bar::<'static, 'static, 'static, u8, i32, i32, false>();

            reuse <u8 as Trait::<'static, 'static, i32>>::foo as bar1;
            //~^ ERROR: type annotations needed [E0284]
            bar1::<'static, u8, i32, true>();
            //~^ ERROR: function takes 4 generic arguments but 3 generic arguments were supplied [E0107]
            //~| ERROR: function takes 3 lifetime arguments but 1 lifetime argument was supplied [E0107]

            reuse <u8 as Trait::<'static, 'static, i32>>::foo::<'static, u32, true> as bar2;
            bar2::<u8>();
            //~^ ERROR: function takes 4 generic arguments but 1 generic argument was supplied [E0107]
        }

        with_ctx::<i32, i32, i32, 1, true>();
    }
}

// Testing lifetimes + types in parent, lifetimes + types/consts in child,
// adding self ty to reuse, testing using generic params from delegation parent
// context, testing predicates inheritance
mod test_11 {
    trait Bound0 {}
    trait Bound1 {}
    trait Bound2<T, U> {}

    trait Trait<'a: 'static, T, P>
    where
        Self: Sized,
        T: Bound0,
        P: Bound2<P, Vec<Vec<Vec<T>>>>,
    {
        fn foo<'d: 'd, U: Bound1, const M: bool>() {}
    }

    impl Bound0 for u32 {}
    impl Bound1 for String {}
    impl<'a: 'static, T: Bound0, P: Bound2<P, Vec<Vec<Vec<T>>>>> Trait<'a, T, P> for usize {}

    struct Struct;
    impl Bound2<Struct, Vec<Vec<Vec<u32>>>> for Struct {}

    pub fn check<'b>() {
        reuse <usize as Trait>::foo;
        //~^ ERROR: missing generics for trait `test_11::Trait` [E0107]
        foo::<'static, 'b, usize, u32, Struct, String, false>();
    }
}

// Testing lifetimes + types/consts in parent with defaults, none in child,
// reuse without user-specified args
mod test_12 {
    trait Trait<'a, T = usize, const N: usize = 123>: Sized {
        fn foo(self) {}
    }

    impl Trait<'static, i32, 1> for u8 {}

    pub fn check() {
        reuse Trait::foo as bar;

        bar::<u8, i32, 1>(123);
    }
}

pub fn main() {
    test_1::check();
    test_2::check();
    test_3::check();
    test_4::check();
    test_5::check();
    test_6::check();
    test_7::check();
    test_8::check();
    test_9::check();
    test_10::check();
    test_11::check();
    test_12::check();
}
