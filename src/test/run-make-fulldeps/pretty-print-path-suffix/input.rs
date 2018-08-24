#![crate_type="lib"]

pub fn
foo() -> i32
{ 45 }

pub fn bar() -> &'static str { "i am not a foo." }

pub mod nest {
    pub fn foo() -> &'static str { "i am a foo." }

    struct S;
    impl S {
        fn foo_method(&self) -> &'static str {
            return "i am very similar to foo.";
        }
    }
}
