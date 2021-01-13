#![warn(clippy::items_after_statements)]

fn ok() {
    fn foo() {
        println!("foo");
    }
    foo();
}

fn last() {
    foo();
    fn foo() {
        println!("foo");
    }
}

fn main() {
    foo();
    fn foo() {
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
