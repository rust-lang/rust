// run-rustfix

#![feature(staged_api)]
#![stable(since = "1.0.0", feature = "test")]
#![deny(deprecated)]
#![allow(dead_code)]

struct Foo;

impl Foo {
    #[rustc_deprecated(
        since = "1.0.0",
        reason = "replaced by `replacement`",
        suggestion = "replacement",
    )]
    #[stable(since = "1.0.0", feature = "test")]
    fn deprecated(&self) {}

    fn replacement(&self) {}
}

mod bar {
    #[rustc_deprecated(
    since = "1.0.0",
    reason = "replaced by `replacement`",
    suggestion = "replacement",
    )]
    #[stable(since = "1.0.0", feature = "test")]
    pub fn deprecated() {}

    pub fn replacement() {}
}

fn main() {
    let foo = Foo;
    foo.deprecated(); //~ ERROR use of deprecated

    bar::deprecated(); //~ ERROR use of deprecated
}
