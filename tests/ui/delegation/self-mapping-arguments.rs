//@ run-pass
//@ check-run-results

#![feature(fn_delegation)]


mod default_test {
    trait MyAdd {
        fn add(self, other: Self) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> usize { self + other }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        assert_eq!(W(1).add(W(2)), W(3))
    }
}

mod arguments_mapping_works_without_return_self {
    trait MyAdd {
        fn add(self, other: Self);
    }

    impl MyAdd for usize {
        fn add(self, other: usize) {
            let result = self + other;
            println!("{result}");
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        W(2).add(W(10));
    }
}

// Test simple mapping: Box<Self> -> Self.
mod default_test_args_adjustments {
    trait MyAdd {
        fn add(self, other: Self);
    }

    impl MyAdd for usize {
        fn add(self, other: Self) { println!("{}", self + other); }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<usize>);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        assert_eq!(W(Box::new(5)).add(W(Box::new(5))), ());
    }
}

// Test a bit more complex mappings:
// `Rc<Arc<Rc<Arc<Rc<usize>>>>>` -> `&Arc<Rc<Self>>`
// `Rc<Arc<Rc<Arc<Rc<usize>>>>>` -> `&Arc<Rc<Arc<Rc<Self>>>>`
// `Rc<Arc<Rc<Arc<Rc<usize>>>>>` -> `&Self`
mod default_test_args_adjustments_2 {
    use std::sync::Arc;
    use std::rc::Rc;

    trait MyAdd {
        fn add(&self, other: &Arc<Rc<Self>>, another_other: &Arc<Rc<Arc<Rc<Self>>>>);
    }

    impl MyAdd for usize {
        fn add(&self, other: &Arc<Rc<Self>>, another_other: &Arc<Rc<Arc<Rc<Self>>>>) {
            println!("{}", *self + ***other + *****another_other);
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Rc<Arc<Rc<Arc<Rc<usize>>>>>);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Rc::new(Arc::new(Rc::new(Arc::new(Rc::new(x))))))
        }
        assert_eq!(
            w(10).add(
                &Arc::new(Rc::new(w(10))),
                &Arc::new(Rc::new(Arc::new(Rc::new(w(10)))))
            ),
            ()
        );
    }
}

// Same as previous test by with more arguments and receiver is complex: `&Arc<Rc<Self>>`.
mod default_test_args_adjustments_3 {
    use std::sync::Arc;
    use std::rc::Rc;

    trait MyAdd {
        fn add(self: &Arc<Rc<Self>>, a: &Arc<Rc<Self>>, b: &Arc<Rc<Arc<Rc<Self>>>>);
    }

    impl MyAdd for usize {
        fn add(self: &Arc<Rc<Self>>, a: &Arc<Rc<Self>>, b: &Arc<Rc<Arc<Rc<Self>>>>) {
            println!("{}", ***self + ***a + *****b);
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Rc<Arc<Rc<Arc<Rc<usize>>>>>);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Rc::new(Arc::new(Rc::new(Arc::new(Rc::new(x))))))
        }
        assert_eq!(
            Arc::new(Rc::new(w(15))).add(
                &Arc::new(Rc::new(w(15))),
                &Arc::new(Rc::new(Arc::new(Rc::new(w(15)))))
            ),
            ()
        );
    }
}

// Test mappings from `Box<Box<Box<usize>>>` to `Self`, `&mut Self`, `&Self`.
mod default_test_args_adjustments_4 {
    trait MyAdd {
        fn add(self: &Self, other: &mut Self, another_other: Self);
    }

    impl MyAdd for usize {
        fn add(self: &Self, other: &mut Self, another_other: Self) {
            println!("{}", *self + *other + another_other);
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<Box<Box<usize>>>);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        fn w(x: usize) -> W {
            W(Box::new(Box::new(Box::new(x))))
        }
        assert_eq!(
            w(20).add(&mut w(20), w(20)),
            ()
        );
    }
}

// Add non-Self args (`a: ()`, `b: impl MyAdd`) to ensure that mapping
// is not applied to them.
mod default_test_with_non_self_args {
    trait MyAdd: std::fmt::Debug {
        fn add(self, a: (), other: Self, b: impl MyAdd) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, a: (), other: usize, b: impl MyAdd) -> usize {
            println!("Non-self arg a: {a:?}");
            println!("Non-self arg b: {b:?}");
            self + other
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);
    reuse impl MyAdd for W {
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        assert_eq!(W(1).add((), W(2), 123), W(3))
    }
}

fn main() {
    println!("default_test");
    default_test::check();

    println!("arguments_mapping_works_without_return_self");
    arguments_mapping_works_without_return_self::check();

    println!("default_test_args_adjustments");
    default_test_args_adjustments::check();

    println!("default_test_args_adjustments_2");
    default_test_args_adjustments_2::check();

    println!("default_test_args_adjustments_3");
    default_test_args_adjustments_3::check();

    println!("default_test_args_adjustments_4");
    default_test_args_adjustments_4::check();

    println!("default_test_with_non_self_args");
    default_test_with_non_self_args::check();
}
