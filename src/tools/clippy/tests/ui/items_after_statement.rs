#![warn(clippy::items_after_statements)]
#![allow(clippy::uninlined_format_args)]

fn ok() {
    fn foo() {
        println!("foo");
    }
    foo();
}

fn last() {
    foo();
    fn foo() {
        //~^ items_after_statements

        println!("foo");
    }
}

fn main() {
    foo();
    fn foo() {
        //~^ items_after_statements

        println!("foo");
    }
    foo();
}

fn mac() {
    let mut a = 5;
    println!("{}", a);
    // do not lint this, because it needs to be after `a`
    macro_rules! b {
        () => {{
            a = 6;
            fn say_something() {
                //~^ items_after_statements
                println!("something");
            }
        }};
    }
    b!();
    println!("{}", a);
}

fn semicolon() {
    struct S {
        a: u32,
    };
    impl S {
        fn new(a: u32) -> Self {
            Self { a }
        }
    }

    let _ = S::new(3);
}

fn item_from_macro() {
    macro_rules! static_assert_size {
        ($ty:ty, $size:expr) => {
            const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
        };
    }

    let _ = 1;
    static_assert_size!(u32, 4);
}

fn allow_attribute() {
    let _ = 1;
    #[allow(clippy::items_after_statements)]
    const _: usize = 1;
}
