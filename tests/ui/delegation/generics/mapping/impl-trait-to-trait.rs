#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing types in parent, types in child reuse,
// testing predicates inheritance,
// with additional generic params in delegation parent
mod test_1 {
    trait Trait0 {}

    trait Trait1<T> {
        fn foo<U>(&self)
        where
            T: Trait0,
            U: Trait0,
        {
        }
    }

    struct F;
    impl<T> Trait1<T> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, T, A, B> Trait1<T> for S<'a, 'b, 'c, A, B> {
        reuse Trait1::<T>::foo { &self.0 }
        //~^ ERROR: type annotations needed [E0283]
    }

    impl Trait0 for u16 {}

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32> as Trait1<u16>>::foo::<u16>(&s);
    }
}

// Testing none in parent, none in child reuse,
// with additional generic params in delegation parent
mod test_2 {
    trait Trait {
        fn foo(&self) {}
    }

    struct F;
    impl Trait for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32, true> as Trait>::foo(&s);
    }
}

// Testing lifetimes + types in parent, none in child reuse,
// with additional generic params in delegation parent
mod test_3 {
    trait Trait<'a, 'b, 'c, X, Y, Z> {
        fn foo(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c, X, Y, Z> Trait<'a, 'b, 'c, X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<'a, 'b, 'static, A, String, bool>
        for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<'a, 'b, 'static, A, String, bool>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32, true>
            as Trait<'static, 'static, 'static, i32, String, bool>>::foo(&s);
    }
}

// Testing lifetimes + types in parent, lifetimes + types/consts in child reuse,
// with additional generic params in delegation parent
mod test_4 {
    trait Trait<'a, 'b, 'c, X, Y, Z> {
        fn foo<'x: 'x, 'y: 'y, 'z: 'z, A, B, C, const XX: usize>(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c, X, Y, Z> Trait<'a, 'b, 'c, X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<'a, 'b, 'static, A, String, bool>
        for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<'a, 'b, 'static, A, String, bool>::foo { &self.0 }
        //~^ ERROR: type annotations needed [E0284]
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32, true>
            as Trait<'static, 'static, 'static, i32, String, bool>>
                ::foo::<'static, 'static, 'static, i32, i32, i32, 1>(&s);
    }
}

// Testing types in parent, lifetimes in child reuse
// with additional generic params in delegation parent
mod test_5 {
    trait Trait<X, Y, Z> {
        fn foo<'a: 'a, 'b: 'b, 'c: 'c>(&self) {}
    }

    struct F;
    impl<X, Y, Z> Trait<X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<A, String, bool> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<A, String, bool>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<i32, String, bool>>
            ::foo::<'static, 'static, 'static>(&s);
        <S::<i32, i32, true> as Trait<i32, String, bool>>::foo(&s);
    }
}

// Testing types in parent, types in child reuse
// with additional generic params in delegation parent
mod test_6 {
    trait Trait<X, Y, Z> {
        fn foo<A, B, C>(&self) {}
    }

    struct F;
    impl<X, Y, Z> Trait<X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<A, String, bool> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<A, String, bool>::foo { &self.0 }
        //~^ ERROR: type annotations needed [E0282]
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<i32, String, bool>>::foo::<i32, i32, i32>(&s);
    }
}

// Testing types in parent, none in child reuse
// with additional generic params in delegation parent
mod test_7 {
    trait Trait<X, Y, Z> {
        fn foo(&self) {}
    }

    struct F;
    impl<X, Y, Z> Trait<X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<A, String, bool> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<A, String, bool>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<i32, String, bool>>::foo(&s);
    }
}

// Testing lifetimes in parent, none in child reuse
// with additional generic params in delegation parent
mod test_8 {
    trait Trait<'a, 'b, 'c> {
        fn foo(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c> Trait<'a, 'b, 'c> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<'a, 'b, 'c> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<'a, 'static, 'b>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<'static, 'static, 'static>>::foo(&s);
    }
}

// Testing lifetimes in parent, lifetimes in child reuse
// with additional generic params in delegation parent
mod test_9 {
    trait Trait<'a, 'b, 'c> {
        fn foo<'x: 'x, 'y: 'y>(&self) {}
    }

    struct F;
    impl<'a, 'b, 'c> Trait<'a, 'b, 'c> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<'a, 'b, 'c> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<'a, 'static, 'b>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<'static, 'static, 'static>>::foo(&s);
        <S::<i32, i32, true> as Trait<'static, 'static, 'static>>
            ::foo::<'static, 'static>(&s);
    }
}

fn main() {
    test_1::check();
    test_2::check();
    test_3::check();
    test_4::check();
    test_5::check();
    test_6::check();
    test_7::check();
    test_8::check();
    test_9::check();
}
