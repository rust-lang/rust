// run-pass
// aux-build:macro-only-syntax.rs

extern crate macro_only_syntax;

#[macro_only_syntax::expect_extern_impl_block]
extern {
    impl T {
        fn f();
    }
}

fn main() {}
