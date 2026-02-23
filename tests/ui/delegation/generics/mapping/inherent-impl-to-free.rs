#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![allow(late_bound_lifetime_arguments)]

//! This is one of the mapping tests, which tests mapping of delegee parent and child
//! generic params, whose main goal is to create cases with
//! different number of lifetimes/types/consts in delegee child and parent; and in
//! delegation parent if applicable. At some tests predicates are
//! added. At some tests user-specified args are specified in reuse statement.

// Testing lifetimes + types/consts OR types/consts OR none in delegation parent,
// lifetimes + types/consts in child reuse,
// with(out) user-specified args
mod test_1 {
    mod to_reuse {
        pub fn foo<'a: 'a, 'b: 'b, A, B, const N: usize>() {}
    }

    struct X1<'a, 'b, T, X, const N: usize>(&'a T, &'b X, &'a [i32; N]);
    impl<'a, 'b, T, E, const N: usize> X1<'a, 'b, T, E, N> {
        reuse to_reuse::foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    struct X2<T, X, const N: usize>(T, X, &'static [i32; N]);
    impl<T, E, const N: usize> X2<T, E, N> {
        reuse to_reuse::foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    struct X3;
    impl X3 {
        reuse to_reuse::foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse to_reuse::foo::<'static, 'static, i32, String, 1> as bar;
    }

    pub fn check() {
        X1::<'static, 'static, i32, i32, 1>
            ::foo::<'static, 'static, String, String, 123>();
        X1::<'static, 'static, i32, i32, 1>::bar();
        //~^ ERROR: type annotations needed [E0284]

        X2::<i32, i32, 1>::foo::<'static, 'static, String, String, 123>();
        X2::<i32, i32, 1>::bar();
        //~^ ERROR: type annotations needed [E0284]

        X3::foo::<'static, 'static, String, String, 123>();
        X3::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing lifetimes + types/consts OR types/consts OR none in parent,
// types/consts in child reuse, with(out) user-specified args
mod test_2 {
    fn foo<A, B, const N: usize>() {}

    struct X1<'a, 'b, T, X, const N: usize>(&'a T, &'b X, &'a [i32; N]);
    impl<'a, 'b, T, E, const N: usize> X1<'a, 'b, T, E, N> {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<i32, String, 1> as bar;
    }

    struct X2<T, X, const N: usize>(T, X, &'static [i32; N]);
    impl<T, E, const N: usize> X2<T, E, N> {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<i32, String, 1> as bar;
    }

    struct X3;
    impl X3 {
        reuse foo;
        //~^ ERROR: type annotations needed [E0284]
        reuse foo::<i32, String, 1> as bar;
    }

    pub fn check() {
        X1::<'static, 'static, i32, i32, 1>::foo::<String, String, 123>();
        X1::<'static, 'static, i32, i32, 1>::bar();
        //~^ ERROR: type annotations needed [E0284]

        X2::<i32, i32, 1>::foo::<String, String, 123>();
        X2::<i32, i32, 1>::bar();
        //~^ ERROR: type annotations needed [E0284]

        X3::foo::<String, String, 123>();
        X3::bar();
        //~^ ERROR: type annotations needed [E0284]
    }
}

// Testing lifetimes + types/consts OR types/consts OR none in parent,
// none in child reuse
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
