// test for issue - https://github.com/rust-lang/rust/issues/161213
macro_rules! m {
    ($value:expr) => {
        enum Foo {
            Bar = $value,
            //~^ error: expected a pattern range bound, found an expression
        }
        fn main() {
            match 0 {
                0..Foo::Bar + $value | _ => (),
            }
        }
    };
}
m!(1);
