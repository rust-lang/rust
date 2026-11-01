//@ run-pass
//@ check-run-results

#![feature(fn_delegation)]

mod simple_self {
    trait MyAdd {
        fn add(self, other: Self) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> usize {
            self + other
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);

    reuse impl MyAdd for W {
        println!("simple_self {self:?}");
        self.0
    }

    pub fn check() {
        assert_eq!(W(1).add(W(2)), W(3))
    }
}

mod box_self {
    trait MyAdd {
        fn add(self, other: Self) -> Box<Self>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Box<usize> {
            Box::new(self + other)
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<usize>);

    reuse impl MyAdd for W {
        println!("box_self {self:?}");
        *self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Box::new(x))
        }

        assert_eq!(w(1).add(w(2)), Box::new(w(3)))
    }
}

mod rc_self {
    use std::rc::Rc;

    trait MyAdd {
        fn add(self, other: Self) -> Rc<Self>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Rc<usize> {
            Rc::new(self + other)
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Rc<usize>);

    reuse impl MyAdd for W {
        println!("rc_self {self:?}");
        *self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Rc::new(x))
        }

        assert_eq!(w(1).add(w(2)), Rc::new(w(3)))
    }
}

mod arc_self {
    use std::sync::Arc;

    trait MyAdd {
        fn add(self, other: Self) -> Arc<Self>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Arc<usize> {
            Arc::new(self + other)
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Arc<usize>);

    reuse impl MyAdd for W {
        println!("arc_self {self:?}");
        *self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Arc::new(x))
        }

        assert_eq!(w(1).add(w(2)), Arc::new(w(3)))
    }
}

mod custom_froms {
    #[derive(Debug)]
    struct S1<A> {
        a: A,
    }

    impl<A> From<A> for S1<A> {
        fn from(a: A) -> S1<A> {
            S1 { a }
        }
    }

    #[derive(Debug)]
    struct S2<T> {
        t: T,
    }

    impl<T> From<T> for S2<T> {
        fn from(t: T) -> S2<T> {
            S2 { t }
        }
    }

    #[derive(Debug)]
    struct S3<'a, const C: usize, T, U, const B: bool> {
        t: T,
        pd: std::marker::PhantomData<&'a [(usize, U); C]>
    }

    impl<'a, const C: usize, T, const B: bool> From<T> for S3<'a, C, T, (), B> {
        fn from(t: T) -> S3<'a, C, T, (), B> {
            S3 {
                t,
                pd: std::marker::PhantomData::<&'a [(usize, ()); C]>,
            }
        }
    }

    trait MyAdd: Sized {
        fn add(self, other: Self) -> S1<S1<S3<'static, 123, S2<S2<S1<Self>>>, (), true>>>;
    }

    fn create_monster_struct<T>(x: T) -> S1<S1<S3<'static, 123, S2<S2<S1<T>>>, (), true>>> {
        S1::from(S1::from(S3::from(S2::from(S2::from(S1::from(x))))))
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> S1<S1<S3<'static, 123, S2<S2<S1<Self>>>, (), true>>> {
            create_monster_struct(self + other)
        }
    }

    #[derive(Debug)]
    struct W(S1<S1<S3<'static, 123, S2<S2<S1<usize>>>, (), true>>>);

    impl From<W> for S1<S1<S3<'static, 123, S2<S2<S1<W>>>, (), true>>> {
        fn from(x: W) -> Self {
            create_monster_struct(x)
        }
    }

    reuse impl MyAdd for W {
        println!("custom_froms {self:?}");
        self.0.a.a.t.t.t.a
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(create_monster_struct(x))
        }

        assert_eq!(w(1).add(w(2)).a.a.t.t.t.a.0.a.a.t.t.t.a, 3)
    }
}

fn main() {
    simple_self::check();
    box_self::check();
    rc_self::check();
    arc_self::check();
    custom_froms::check();
}
