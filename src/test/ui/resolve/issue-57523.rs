// build-pass (FIXME(62277): could be check-pass?)

struct S(u8);

impl S {
    fn method1() -> Self {
        Self(0)
    }
}

macro_rules! define_method { () => {
    impl S {
        fn method2() -> Self {
            Self(0) // OK
        }
    }
}}

define_method!();

fn main() {}
