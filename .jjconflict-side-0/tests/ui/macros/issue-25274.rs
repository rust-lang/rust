//@ run-pass

macro_rules! test {
    (
        fn fun() -> Option<Box<$t:ty>>;
    ) => {
        fn fun(x: $t) -> Option<Box<$t>>
        { Some(Box::new(x)) }
    }
}

test! {
    fn fun() -> Option<Box<i32>>;
}

fn main() {
    println!("{}", fun(0).unwrap());
}
