//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod infer_types {
    mod to_reuse {
        pub fn foo<T, U>(x: T, y: U) -> (T, U) { (x, y) }
        pub fn bar<T>(x: T) -> T { x }
    }

    pub fn check_shallow_inf_vars() {
        #[derive(PartialEq, Debug, Copy, Clone)]
        struct T;
        #[derive(PartialEq, Debug, Copy, Clone)]
        struct U;
        {
            reuse to_reuse::foo::<_, _>;
            assert_eq!(foo(T, U), (T, U));
        }
        {
            reuse to_reuse::foo;
            assert_eq!(foo(T, U), (T, U));
        }
        {
            reuse to_reuse::foo::<U, _>;
            assert_eq!(foo(U, 0), (U, 0));
        }
    }

    pub fn check_deep_inf_vars() {
        #[derive(PartialEq, Debug, Copy, Clone)]
        struct Us;

        #[derive(PartialEq, Debug, Copy, Clone)]
        struct Ss<A, B> {
            x: A,
            y: B,
        }
        let x = Ss::<u8, [i32; 1]> { x: 0, y: [1] };
        {
            reuse to_reuse::foo::<Ss<u8, _>, _>;
            let res = foo(x, Us);
            assert_eq!(res.1, Us);
            assert_eq!(res.0, x);
        }
        {
            reuse to_reuse::foo;
            let res = foo(x, Us);
            assert_eq!(res.1, Us);
            assert_eq!(res.0, x);
        }
    }

    pub fn check_type_aliases() {
        trait Trait<T> {
            type Type;
        }

        impl<T> Trait<T> for u8 {
            type Type = ();
        }

        type Type = ();

        {
            reuse to_reuse::bar::<<u8 as Trait<u8>>::Type>;
            assert_eq!(bar(()), ());
        }
        {
            reuse to_reuse::bar::<Type>;
            assert_eq!(bar(()), ());
        }
    }
}

mod infer_late_bound_regions {
    mod to_reuse {
        pub fn foo<T>(x: &T) -> &T { x }
    }

    pub fn check() {
        let x = 1;
        {
            reuse to_reuse::foo;
            assert_eq!(*foo(&x), 1);
        }
    }
}

mod infer_early_bound_regions {
    mod to_reuse {
        pub fn foo<'a: 'a, T>(x: &'a T) -> &'a T { x }
    }

    pub fn check_shallow_inf_vars() {
        let x = 1;
        {
            reuse to_reuse::foo::<'_, _>;
            assert_eq!(*foo(&x), 1);
        }
        {
            reuse to_reuse::foo;
            assert_eq!(*foo(&x), 1);
        }
    }

    pub fn check_deep_inf_vars() {
        #[derive(PartialEq, Debug, Copy, Clone)]
        struct S<'a, U> {
            x: &'a U
        }
        let x = 0;
        let s = S { x: &x };
        {
            reuse to_reuse::foo::<'_, S<'_, i32>>;
            assert_eq!(*foo(&s), s);
        }
        {
            reuse to_reuse::foo;
            assert_eq!(*foo(&s), s);
        }
    }
}

mod constants {
    mod to_reuse {
        pub fn foo1<const N: i32>() -> i32 { N }
        pub fn foo2<T>(x: T) -> T { x }
    }

    pub fn check() {
        {
            reuse to_reuse::foo1::<42>;
            assert_eq!(foo1(), 42);
        }

        #[derive(PartialEq, Debug)]
        struct S<const N: i32 = 1>;
        {
            reuse to_reuse::foo2::<S>;
            let s = S;
            assert_eq!(foo2(s), S::<1>);
        }
        {
            reuse to_reuse::foo2::<S::<2>>;
            let s = S;
            assert_eq!(foo2(s), S::<2>);
        }
    }
}

fn main() {
    infer_types::check_shallow_inf_vars();
    infer_types::check_deep_inf_vars();
    infer_types::check_type_aliases();
    infer_late_bound_regions::check();
    infer_early_bound_regions::check_shallow_inf_vars();
    infer_early_bound_regions::check_deep_inf_vars();
    constants::check();
}
