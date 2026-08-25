//@ edition:2018

fn main() {}

fn a_function() -> u32 {
    let x: Option<u32> = None;
    x?; //~ ERROR the `?` operator
    22
}

fn a_closure() -> u32 {
    let a_closure = || {
        let x: Option<u32> = None;
        x?; //~ ERROR the `?` operator
        22
    };
    a_closure()
}

fn a_method() -> u32 {
    struct S;

    impl S {
        fn a_method() {
            let x: Option<u32> = None;
            x?; //~ ERROR the `?` operator
        }
    }

    S::a_method();
    22
}

fn a_trait_method() -> u32 {
    struct S;
    trait T {
        fn a_trait_method() {
            let x: Option<u32> = None;
            x?; //~ ERROR the `?` operator
        }
    }

    impl T for S { }

    S::a_trait_method();
    22
}
