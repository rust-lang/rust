//! Regression test for https://github.com/rust-lang/rust/issues/134146.
//! Traits declared inside bodies cannot be imported through their enclosing function or const.

#![allow(dead_code)]

fn main() {
    {
        trait Hello {
            fn hello(&self) {
                println!("hello world");
            }
        }
        impl<T> Hello for T {}
    }

    ().hello();
    //~^ ERROR no method named `hello` found
}

fn nested_module() {
    mod inner {
        pub trait Nested {
            fn nested(&self) {}
        }
        impl Nested for () {}
    }
}

fn use_nested() {
    ().nested();
    //~^ ERROR no method named `nested` found
}

const _: () = {
    trait InConst {
        fn in_const(&self) {}
    }
    impl InConst for () {}
};

fn use_const() {
    ().in_const();
    //~^ ERROR no method named `in_const` found
}

fn multiple_candidates() {
    {
        trait First {
            fn several(&self) {}
        }
        trait Second {
            fn several(&self) {}
        }
        impl First for () {}
        impl Second for () {}
    }
    ().several();
    //~^ ERROR no method named `several` found
}
