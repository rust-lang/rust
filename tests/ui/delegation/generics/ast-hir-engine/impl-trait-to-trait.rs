//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

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
    }

    impl Trait0 for u16 {}

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32> as Trait1<u16>>::foo::<u16>(&s);
    }
}

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
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<'static, 'static, 'static, i32, i32, true>
            as Trait<'static, 'static, 'static, i32, String, bool>>
                ::foo::<'static, 'static, 'static, i32, i32, i32, 1>(&s);
    }
}

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

mod test_6 {
    trait Trait<X, Y, Z> {
        fn foo<A, B, C>(&self) {}
    }

    struct F;
    impl<X, Y, Z> Trait<X, Y, Z> for F {}

    struct S<'a, 'b, 'c, A, B, const C: bool>(F, &'a A, &'b B, &'c B);

    impl<'a, 'b, 'c, A, B, const C: bool> Trait<A, String, bool> for S<'a, 'b, 'c, A, B, C> {
        reuse Trait::<A, String, bool>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        <S::<i32, i32, true> as Trait<i32, String, bool>>::foo::<i32, i32, i32>(&s);
    }
}

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
