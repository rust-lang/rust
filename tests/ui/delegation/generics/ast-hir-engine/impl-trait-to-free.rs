//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod test_1 {
    mod to_reuse {
        pub fn foo<'a: 'a, 'b: 'b, A, B, const N: usize>() {}
        pub fn bar<'a: 'a, 'b: 'b, A, B, const N: usize>(x: &super::XX) {}
    }

    trait Trait<'a, 'b, 'c, A, B, const N: usize>: Sized {
        fn foo<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>(&self) {}
        fn oof() {}
        fn rab(&self) {}
    }

    struct X<'x1, 'x2, 'x3, 'x4, X1, X2, const X3: usize>(
        &'x1 X1, &'x2 X2, &'x3 X1, &'x4 [usize; X3]);
    type XX = X::<'static, 'static, 'static, 'static, i32, i32, 3>;

    impl<'a, 'b, 'c, A, B, const N: usize> Trait<'a, 'b, 'c, A, B, N> for XX {
        reuse to_reuse::foo;
        reuse to_reuse::bar;

        reuse to_reuse::foo::<'a, 'c, A, String, 322> as oof;
        reuse to_reuse::bar::<'a, 'c, i32, B, 223> as rab;
    }

    pub fn check() {
        let x = X(&1, &2, &3, &[1, 2, 3]);

        <XX as Trait<'static, 'static, 'static, i32, i32, 1>>
            ::foo::<'static, 'static, i8, i16, 123>();
        <XX as Trait<'static, 'static, 'static, i32, i32, 1>>
            ::bar::<'static, 'static, String, i16, 123>(&x);
        <XX as Trait<'static, 'static, 'static, i32, i32, 1>>::oof();
        <XX as Trait<'static, 'static, 'static, i32, i32, 1>>::rab(&x);
    }
}

mod test_2 {
    mod to_reuse {
        pub fn foo<A, B, const N: usize>() {}
        pub fn bar<A, B, const N: usize>(x: &super::X) {}
    }

    trait Trait<'a, 'b, 'c, A, B, const N: usize>: Sized {
        fn foo<AA, BB, const NN: usize>() {}
        fn bar<AA, BB, const NN: usize>(&self) {}
        fn oof() {}
        fn rab(&self) {}
    }

    struct X;
    impl<'a, A, B, const N: usize> Trait<'a, 'static, 'static, A, B, N> for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;

        reuse to_reuse::foo::<A, String, 322> as oof;
        reuse to_reuse::bar::<i32, B, 223> as rab;
    }

    pub fn check() {
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::foo::<i8, i16, 123>();
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::bar::<X, i16, 123>(&X);
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::oof();
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::rab(&X);
    }
}

mod test_3 {
    mod to_reuse {
        pub fn foo() {}
        pub fn bar(x: &super::X) {}
    }

    trait Trait<'a, 'b, 'c, A, B, const N: usize>: Sized {
        fn foo() {}
        fn bar(&self) {}
    }

    struct X;
    impl<'a, A, B, const N: usize> Trait<'a, 'static, 'static, A, B, N> for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;
    }

    pub fn check() {
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::foo();
        <X as Trait<'static, 'static, 'static, i32, i32, 1>>::bar(&X);
    }
}

mod test_4 {
    mod to_reuse {
        pub fn foo<'a: 'a, 'b: 'b, A, B, const N: usize>() {}
        pub fn bar<'a: 'a, 'b: 'b, A, B, const N: usize>(x: &super::X) {}
    }

    trait Trait<A, B, const N: usize>: Sized {
        fn foo<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>(&self) {}
        fn oof() {}
        fn rab(&self) {}
    }

    struct X;
    impl<'a, 'c, A, B, const N: usize> Trait<A, B, N> for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;

        reuse to_reuse::foo::<'a, 'c, A, String, 322> as oof;
        reuse to_reuse::bar::<'a, 'c, i32, B, 223> as rab;
    }

    pub fn check() {
        <X as Trait<i32, i32, 1>>::foo::<'static, 'static, i8, i16, 123>();
        <X as Trait<i32, i32, 1>>::bar::<'static, 'static, X, i16, 123>(&X);
        <X as Trait<i32, i32, 1>>::oof();
        <X as Trait<i32, i32, 1>>::rab(&X);
    }
}

mod test_5 {
    mod to_reuse {
        pub fn foo<'a: 'a, 'b: 'b, A, B, const N: usize>() {}
        pub fn bar<'a: 'a, 'b: 'b, A, B, const N: usize>(x: &super::X::<A, B>) {}
    }

    trait Trait: Sized {
        fn foo<'x: 'x, 'y: 'y, AA, BB, const NN: usize>() {}
        fn bar<'x: 'x, 'y: 'y, AA, BB, const NN: usize>(&self) {}
        fn oof() {}
        fn rab(&self) {}
    }

    struct X<A, B>(A, B);
    impl<'a, 'c, A, B> Trait for X<A, B> {
        reuse to_reuse::foo::<'a, 'c, A, B, 322> as oof;
        reuse to_reuse::bar::<'a, 'c, A, B, 223> as rab;
    }

    pub fn check() {
        <X::<i32, i32> as Trait>::oof();
        <X::<i32, i32> as Trait>::rab(&X(1, 2));
    }
}

mod test_6 {
    mod to_reuse {
        pub fn foo<A, B, const N: usize>() {}
        pub fn bar<A, B, const N: usize>(x: &super::X) {}
    }

    trait Trait<A, B, const N: usize>: Sized {
        fn foo<AA, BB, const NN: usize>() {}
        fn bar<AA, BB, const NN: usize>(&self) {}
        fn oof() {}
        fn rab(&self) {}
    }

    struct X;
    impl<'a, 'c, A, B, const N: usize> Trait<A, B, N> for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;

        reuse to_reuse::foo::<A, String, 322> as oof;
        reuse to_reuse::bar::<i32, B, 223> as rab;
    }

    pub fn check() {
        <X as Trait<i32, i32, 1>>::foo::<i8, i16, 123>();
        <X as Trait<i32, i32, 1>>::bar::<X, i16, 123>(&X);
        <X as Trait<i32, i32, 1>>::oof();
        <X as Trait<i32, i32, 1>>::rab(&X);
    }
}

mod test_7 {
    mod to_reuse {
        pub fn foo() {}
        pub fn bar(x: &super::X) {}
    }

    trait Trait<A, B, const N: usize>: Sized {
        fn foo() {}
        fn bar(&self) {}
    }

    struct X;
    impl<'a, 'c, A, B, const N: usize> Trait<A, B, N> for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;
    }

    pub fn check() {
        <X as Trait<i32, i32, 1>>::foo();
        <X as Trait<i32, i32, 1>>::bar(&X);
    }
}

mod test_8 {
    mod to_reuse {
        pub fn foo() {}
        pub fn bar(x: &super::X) {}
    }

    trait Trait: Sized {
        fn foo() {}
        fn bar(&self) {}
    }

    struct X;
    impl Trait for X {
        reuse to_reuse::foo;
        reuse to_reuse::bar;
    }

    pub fn check() {
        <X as Trait>::foo();
        <X as Trait>::bar(&X);
        X::foo();
        X::bar(&X);
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
}
