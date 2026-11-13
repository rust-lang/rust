#![feature(fn_delegation)]

mod target_expr_doesnt_relower_when_defs_inside {
    trait MyAdd {
        fn add(self, other: Self) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> usize { self + other }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);
    reuse impl MyAdd for W {
    //~^ ERROR: attempted to lower target expression with definitions more than once while mapping argument
    //~| ERROR: method `add` has a `self` declaration in the trait, but not in the impl
    //~| ERROR: the trait bound `(): target_expr_doesnt_relower_when_defs_inside::MyAdd` is not satisfied
    //~| ERROR: this function takes 2 arguments but 1 argument was supplied
        println!("{self:?}");
        fn foo() {
            println!("hello");
        }

        reuse foo as bar;
        bar();
        bar();

        self.0
    }
}

mod complex_Self_doesnt_map {
    trait MyAdd {
        fn add(self, other: Box<Self>) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, other: Box<usize>) -> usize { self + *other.as_ref() }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(usize);
    reuse impl MyAdd for W { self.0 }
    //~^ ERROR: mismatched types
}

// FIXME(fn_delegation): support heuristics to wrap return value
// with complex type (`Box<Rc<Self>>`).
mod return_type_self_mapping_produces_error {
    trait MyAdd {
        fn add(self, other: Self) -> Self;
    }

    impl MyAdd for usize {
        fn add(self, other: Self) -> Self { self + other }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<usize>);
    reuse impl MyAdd for W { //~ ERROR: mismatched types
        println!("{self:?}");
        let _x = 213;

        self.0
    }

    pub fn check() {
        assert_eq!(W(Box::new(5)).add(W(Box::new(5))), W(Box::new(10)));
    }
}

fn main() {}
