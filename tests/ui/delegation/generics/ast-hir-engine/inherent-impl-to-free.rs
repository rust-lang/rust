//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(warnings)]

mod test_1 {
    mod to_reuse {
        pub fn foo<'a: 'a, 'b: 'b, A, B, const N: usize>() {}
    }

    struct X1<'a, 'b, T, X, const N: usize>(&'a T, &'b X, &'a [i32; N]);
    impl<'a, 'b, T, E, const N: usize> X1<'a, 'b, T, E, N> {
        reuse to_reuse::foo;
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    struct X2<T, X, const N: usize>(T, X, &'static [i32; N]);
    impl<T, E, const N: usize> X2<T, E, N> {
        reuse to_reuse::foo;
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    struct X3;
    impl X3 {
        reuse to_reuse::foo;
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    pub fn check() {
        X1::<'static, 'static, i32, i32, 1>
            ::foo::<'static, 'static, String, String, 123>();
        X1::<'static, 'static, i32, i32, 1>::bar();

        X2::<i32, i32, 1>::foo::<'static, 'static, String, String, 123>();
        X2::<i32, i32, 1>::bar();

        X3::foo::<'static, 'static, String, String, 123>();
        X3::bar();
    }
}

mod test_2 {
    fn foo<A, B, const N: usize>() {}

    struct X1<'a, 'b, T, X, const N: usize>(&'a T, &'b X, &'a [i32; N]);
    impl<'a, 'b, T, E, const N: usize> X1<'a, 'b, T, E, N> {
        reuse foo;
        reuse foo::<i32, String, 1> as bar;
    }

    struct X2<T, X, const N: usize>(T, X, &'static [i32; N]);
    impl<T, E, const N: usize> X2<T, E, N> {
        reuse foo;
        reuse foo::<i32, String, 1> as bar;
    }

    struct X3;
    impl X3 {
        reuse foo;
        reuse foo::<i32, String, 1> as bar;
    }

    pub fn check() {
        X1::<'static, 'static, i32, i32, 1>::foo::<String, String, 123>();
        X1::<'static, 'static, i32, i32, 1>::bar();

        X2::<i32, i32, 1>::foo::<String, String, 123>();
        X2::<i32, i32, 1>::bar();

        X3::foo::<String, String, 123>();
        X3::bar();
    }
}

mod test_3 {
    fn foo() {}

    struct X1<'a, 'b, T, X, const N: usize>(&'a T, &'b X, &'a [i32; N]);
    impl<'a, 'b, T, E, const N: usize> X1<'a, 'b, T, E, N> {
        reuse foo;
    }

    struct X2<T, X, const N: usize>(T, X, &'static [i32; N]);
    impl<T, E, const N: usize> X2<T, E, N> {
        reuse foo;
    }

    struct X3;
    impl X3 {
        reuse foo;
    }

    pub fn check() {
        X1::<'static, 'static, i32, i32, 1>::foo();

        X2::<i32, i32, 1>::foo();

        X3::foo();
    }
}

fn main() {
    test_1::check();
    test_2::check();
    test_3::check();
}
