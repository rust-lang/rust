//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod test_1 {
    trait Trait<T> {
        fn foo(&self) {}
    }

    struct F;
    impl<T> Trait<T> for F {}

    struct S<'a, 'b, 'c, A, B>(F, &'a A, &'b B, &'c B);
    impl<'a, 'b, 'c, A, B> S<'a, 'b, 'c, A, B> {
        reuse Trait::<String>::foo { &self.0 }
    }

    pub fn check() {
        let s = S(F, &123, &123, &123);
        S::<'static, 'static, 'static, i32, i32>::foo(&s);
        s.foo();
    }
}

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
        s.foo();
    }
}

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
        s.foo();
    }
}

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
        s.foo();
    }
}

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
        s.foo();
    }
}

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
