// rustfmt-edition: 2018

#![feature(ergonomic_clones)]

mod closure_syntax {
    fn ergonomic_clone_closure_no_captures() -> i32 {
        let cl = use || {
            1
        };
        cl()
    }

    fn ergonomic_clone_closure_move() -> String {
        let s = String::from("hi");

        let cl = use || {
            s
        };
        cl()
    }

    fn ergonomic_clone_closure_use_cloned() -> Foo {
        let f = Foo;

        let f1 = use || {
            f
        };

        let f2 = use || {
            f
        };

        f
    }

    fn ergonomic_clone_closure_copy() -> i32 {
        let i = 1;

        let i1 = use || {
            i
        };

        let i2 = use || {
            i
        };

        i
    }

    fn nested() {
    let a = Box::new(Foo);
    foo(use || { foo(use || { work(a) }) });
    let x = use || { use || { Foo } };
    let _y = x();
    }

    fn with_binders() {
        for<'a> use || -> () {};
    }
}

mod dotuse {
    fn basic_test(x: i32) -> i32 {
        x.use
            .use
            .abs()
    }

    fn do_not_move_test(x: Foo) -> Foo {
        let s = x.use;
        x
    }
}
