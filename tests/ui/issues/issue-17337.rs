#![feature(staged_api)]
#![deny(deprecated)]

#![unstable(feature = "unstable_test_feature", issue = "none")]

struct Foo;

impl Foo {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[deprecated(since = "1.0.0", note = "text")]
    fn foo(self) {}
}

fn main() {
    Foo
    .foo(); //~ ERROR use of deprecated
}
