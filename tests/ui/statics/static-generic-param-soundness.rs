//! Originally, inner statics in generic functions were generated only once, causing the same
//! static to be shared across all generic instantiations. This created a soundness hole where
//! different types could be coerced through thread-local storage in safe code.
//!
//! This test checks that generic parameters from outer scopes cannot be used in inner statics,
//! preventing this soundness issue.
//!
//! See https://github.com/rust-lang/rust/issues/9186

enum Bar<T> {
    //~^ ERROR parameter `T` is never used
    What,
}

fn foo<T>() {
    static a: Bar<T> = Bar::What;
    //~^ ERROR can't use generic parameters from outer item
}

fn main() {}
