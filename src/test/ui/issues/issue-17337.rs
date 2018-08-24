#![feature(staged_api)]
#![deny(deprecated)]

#![unstable(feature = "unstable_test_feature", issue = "0")]

struct Foo;

impl Foo {
    #[unstable(feature = "unstable_test_feature", issue = "0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn foo(self) {}
}

fn main() {
    Foo
    .foo(); //~ ERROR use of deprecated item
}
