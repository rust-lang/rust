#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing lifetimes + types in parent,
// lifetimes + types/consts in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_1 {
    trait Trait<'b, 'c, 'a, T>: Sized {
        fn foo<'d: 'd, U, const M: bool>(&self) {}
    }

    impl<'b, 'c, 'a, T> Trait<'b, 'c, 'a, T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false> as bar2 {
                Self::get()
        }

        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar4 { self.get_self() }

        // FIXME(fn_delegation): Uncomment those tests when proper support for
        // generics when method call is generated is added

        // reuse Trait::foo::<'static, String, false> as bar5 { Self::get() }
        // reuse Trait::foo as bar6 { Self::get() }
        // reuse Trait::foo::<'static, String, false> as bar7 { self.get_self() }
        // reuse Trait::foo as bar8 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar2 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar4 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar2 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar4 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar2 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<'static, String, false>
            as bar4 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar1::<'static, String, true>(&123);
            //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
            //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, String, true>(&123);
            //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
            //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        //~| ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
    }
}

// Testing types in parent,
// lifetimes + types/consts in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_2 {
    trait Trait<T>: Sized {
        fn foo<'d: 'd, U, const M: bool>(&self) {}
    }

    impl<T> Trait<T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<i32>::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar1::<'static, String, true>(&123);
            //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait3>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait5<i32, u64, String>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, String, true>(&123);
            //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait3>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
    }
}

// Testing lifetimes in parent,
// lifetimes + types/consts in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_3 {
    trait Trait<'b, 'c, 'a>: Sized {
        fn foo<'d: 'd, U, const M: bool>(&self) {}
    }

    impl<'b, 'c, 'a> Trait<'b, 'c, 'a> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false>
            as bar2 { Self::get() }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static>::foo::<'static, String, false>
            as bar4 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar1::<'static, String, true>(&123);
            //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar1::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar1::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, String, true>(&123);
            //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar3::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, String, true>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
    }
}

// Testing none in parent,
// lifetimes + types/consts in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_4 {
    trait Trait: Sized {
        fn foo<'d: 'd, U, const M: bool>(&self) {}
    }

    impl Trait for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar2 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::foo::<'static, String, false> as bar4 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar1::<'static, String, true>(&123);
        <u32 as Trait3>::bar1::<'static, String, true>(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'static, String, true>(&123);
        <u32 as Trait5<i32, u64, String>>::bar1::<'static, String, true>(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait3>::bar2(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait5<i32, u64, String>>::bar2(&123);
        //~^ ERROR: type annotations needed [E0284]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, String, true>(&123);
        <u32 as Trait3>::bar3::<'static, String, true>(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, String, true>(&123);
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, String, true>(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait3>::bar4(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        //~^ ERROR: type annotations needed [E0284]
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing lifetimes + types in parent,
// types/consts in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_5 {
    trait Trait<'b, 'c, 'a, T>: Sized {
        fn foo<U, const M: bool>(&self) {}
    }

    impl<'b, 'c, 'a, T> Trait<'b, 'c, 'a, T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar4 {
            self.get_self()
        }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
        //~^ ERROR: type annotations needed [E0284]
        reuse Trait::<'static, 'static, 'static, i32>::foo::<String, false> as bar4 {
            self.get_self()
        }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar1::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait3>::bar1::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait5<i32, u64, String>>::bar1::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar3::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait3>::bar3::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]
        <u32 as Trait5<i32, u64, String>>::bar3::<String, true>(&123);
        //~^ ERROR: method takes 3 generic arguments but 2 generic arguments were supplied [E0107]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
    }
}

// Testing lifetimes + types in parent,
// none in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_6 {
    trait Trait<'b, 'c, 'a, T>: Sized {
        fn foo(&self) {}
    }

    impl<'b, 'c, 'a, T> Trait<'b, 'c, 'a, T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static, i32>::foo as bar3 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar1(&123);
        //~^ ERROR: type annotations needed [E0282]
        <u32 as Trait3>::bar1(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar1(&123);
        <u32 as Trait5<i32, u64, String>>::bar1(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar3(&123);
        <u32 as Trait3>::bar3(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar3(&123);
        <u32 as Trait5<i32, u64, String>>::bar3(&123);
    }
}

// Testing types in parent,
// none in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with user-specified args, with different target expr
mod test_7 {
    trait Trait<T>: Sized {
        fn foo(&self) {}
    }

    impl<T> Trait<T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar1(&123);
        //~^ ERROR: type annotations needed [E0282]
        <u32 as Trait3>::bar1(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar1(&123);
        <u32 as Trait5<i32, u64, String>>::bar1(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar3(&123);
        <u32 as Trait3>::bar3(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar3(&123);
        <u32 as Trait5<i32, u64, String>>::bar3(&123);
    }
}

// Testing none in parent,
// none in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// without user-specified args, with different target expr
mod test_8 {
    trait Trait: Sized {
        fn foo(&self) {}
    }

    impl Trait for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::foo as bar1 { Self::get() }
        reuse Trait::foo as bar3 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar1(&123);
        <u32 as Trait3>::bar1(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar1(&123);
        <u32 as Trait5<i32, u64, String>>::bar1(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar3(&123);
        <u32 as Trait3>::bar3(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar3(&123);
        <u32 as Trait5<i32, u64, String>>::bar3(&123);
    }
}

// Testing types in parent,
// lifetimes in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_9 {
    trait Trait<T>: Sized {
        fn foo<'a: 'a, 'b: 'b>(&self) {}
    }

    impl<T> Trait<T> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<i32>::foo as bar1 { Self::get() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<i32>::foo as bar3 { self.get_self() }
        reuse Trait::<i32>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
        //~^ ERROR: type annotations needed [E0282]
            ::bar1::<'static, 'static>(&123);
        <u32 as Trait3>::bar1::<'static, 'static>(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'a, 'a>(&123);
        <u32 as Trait5<i32, u64, String>>::bar1::<'a, 'a>(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, 'static>(&123);
        <u32 as Trait3>::bar3::<'static, 'static>(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, 'static>(&123);
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, 'static>(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
    }
}

// Testing lifetimes in parent,
// lifetimes in child,
// in delegation parent with:
// lifetimes + types OR none OR lifetimes OR types,
// with(out) user-specified args, with different target expr
mod test_10 {
    trait Trait<'x, 'y, 'z>: Sized {
        fn foo<'a: 'a, 'b: 'b>(&self) {}
    }

    impl<'x, 'y, 'z> Trait<'x, 'y, 'z> for u8 {}

    trait Trait2<'a, 'b, 'c, X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'a, 'b, 'static>::foo as bar1 { Self::get() }
        reuse Trait::<'a, 'b, 'static>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<'a, 'b, 'static>::foo as bar3 { self.get_self() }
        reuse Trait::<'a, 'b, 'static>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    trait Trait3 {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static>::foo::<'static, 'static> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        reuse Trait::<'static, 'static, 'static>::foo::<'static, 'static> as bar4 {
                self.get_self()
        }
    }

    trait Trait4<'a, 'b, 'c> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'a, 'b, 'static>::foo as bar1 { Self::get() }
        reuse Trait::<'a, 'b, 'static>::foo::<'static, 'static> as bar2 { Self::get() }
        reuse Trait::<'a, 'b, 'static>::foo as bar3 { self.get_self() }
        reuse Trait::<'a, 'b, 'static>::foo::<'static, 'static> as bar4 { self.get_self() }
    }

    trait Trait5<X, Y, Z> {
        fn get() -> &'static u8 { &0 }
        fn get_self(&self) -> &'static u8 { &0 }
        reuse Trait::<'static, 'static, 'static>::foo as bar1 { Self::get() }
        reuse Trait::<'static, 'static, 'static>::foo::<'static, 'static> as bar2 {
            Self::get()
        }
        reuse Trait::<'static, 'static, 'static>::foo as bar3 { self.get_self() }
        reuse Trait::<'static, 'static, 'static>::foo::<'static, 'static> as bar4 {
            self.get_self()
        }
    }

    impl<'a, 'b, 'c, X, Y, Z> Trait2<'a, 'b, 'c, X, Y, Z> for u32 {}
    impl Trait3 for u32 {}
    impl<'a, 'b, 'c> Trait4<'a, 'b, 'c> for u32 {}
    impl<X, Y, Z> Trait5<X, Y, Z> for u32 {}

    pub fn check<'a: 'a>() {
        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar1::<'static, 'static>(&123);
            //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar1::<'static, 'static>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar1::<'a, 'a>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar1::<'a, 'a>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar2(&123);
        <u32 as Trait3>::bar2(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar2(&123);
        <u32 as Trait5<i32, u64, String>>::bar2(&123);

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>
            ::bar3::<'static, 'static>(&123);
            //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait3>::bar3::<'static, 'static>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait4<'a, 'a, 'static>>::bar3::<'static, 'static>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
        <u32 as Trait5<i32, u64, String>>::bar3::<'static, 'static>(&123);
        //~^ ERROR: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]

        <u32 as Trait2<'static, 'static, 'static, i32, i32, i32>>::bar4(&123);
        <u32 as Trait3>::bar4(&123);
        <u32 as Trait4<'a, 'a, 'static>>::bar4(&123);
        <u32 as Trait5<i32, u64, String>>::bar4(&123);
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
}
