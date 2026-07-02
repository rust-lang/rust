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

fn main() {
    default_test::check();
    arguments_mapping_works_without_return_self::check();
}
