#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing types in parent, none in child,
// user-specified args in parent, checking predicates inheritance,
// with additional generic params in delegation parent
mod test_1 {
    trait Trait<T: ToString> {
        fn foo(&self) {}
    }

    struct F;
    impl<T: ToString> Trait<T> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<String>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0283]
        s.foo();
    }
}

// Testing lifetimes + types/consts in parent, none in child,
// with additional generic params in delegation parent
mod test_2 {
    trait Trait<'x, 'y, T, const B: bool> {
        fn foo(&self) {}
    }

    struct F;
    impl<'x, 'y, T, const B: bool> Trait<'x, 'y, T, B> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<'a, 'b, String, true>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0284]
        s.foo();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing lifetimes in parent, none in child,
// with additional generic params in delegation parent
mod test_3 {
    trait Trait<'x, 'y> {
        fn foo(&self) {}
    }

    struct F;
    impl<'x, 'y> Trait<'x, 'y> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<'a, 'b>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        s.foo();
    }
}

// Testing none in parent, none in child,
// with additional generic params in delegation parent
mod test_4 {
    trait Trait {
        fn foo(&self) {}
    }

    struct F;
    impl Trait for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        s.foo();
    }
}

// Testing none in parent, lifetimes in child,
// with additional generic params in delegation parent
mod test_5 {
    trait Trait {
        fn foo<'a: 'a, 'b: 'b>(&self) {}
    }

    struct F;
    impl Trait for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::foo::<'a, 'b> { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        s.foo();
    }
}

// Testing none in parent, lifetimes + types in child,
// with additional generic params in delegation parent
mod test_6 {
    trait Trait {
        fn foo<'a: 'a, 'b: 'b, A, B, C>(&self) {}
    }

    struct F;
    impl Trait for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::foo::<'a, 'b, A, B, String> { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0282]
        s.foo();
    }
}

// Testing lifetimes in parent, lifetimes + types in child,
// with additional generic params in delegation parent
mod test_7 {
    trait Trait<'x, 'y, 'z> {
        fn foo<'a: 'a, 'b: 'b, A, B, C>(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c> Trait<'a, 'b, 'c> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<'a, 'b, 'c>::foo::<'a, 'b, A, B, String> { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0282]
        s.foo();
    }
}

// Testing lifetimes + types in parent, lifetimes + types in child,
// with additional generic params in delegation parent
mod test_8 {
    trait Trait<'x, 'y, 'z, X, Y, Z> {
        fn foo<'a: 'a, 'b: 'b, A, B, C>(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c, X, Y, Z> Trait<'a, 'b, 'c, X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<'a, 'b, 'c, B, A, i32>::foo::<'a, 'b, A, B, String> { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0282]
        s.foo();
    }
}

// Testing lifetimes + types in parent, lifetimes + types in child,
// with additional generic params in delegation parent,
// inside a function with generic params
mod test_9 {
    trait Trait<'x, 'y, 'z, X, Y, Z> {
        fn foo<'a: 'a, 'b: 'b, A, B, C>(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c, X, Y, Z> Trait<'a, 'b, 'c, X, Y, Z> for F {}

    pub fn check<T, U>() {
        struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
        impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
            reuse Trait::<'a, 'b, 'c, B, A, i32>::foo::<'a, 'b, A, B, String> { &self.0 }
        }

        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        //~^ ERROR: type annotations needed [E0282]
        s.foo();
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
    test_9::check::<u32, String>();
}
