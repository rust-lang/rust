#![feature(staged_api)]
#![deny(deprecated)]

#![unstable(feature = "unstable_test_feature")]

struct Foo;

impl Foo {
    #[unstable(feature = "unstable_test_feature")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn foo(self) {}
}

fn main() {
    Foo
    .foo(); //~ ERROR use of deprecated item
}
