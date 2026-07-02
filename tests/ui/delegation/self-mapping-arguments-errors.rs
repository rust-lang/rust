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
    reuse impl MyAdd for W { //~ ERROR: mismatched types
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

fn main() {}
