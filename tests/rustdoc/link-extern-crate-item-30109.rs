//@ build-aux-docs
//@ aux-build:issue-30109-1.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/30109
#![crate_name="issue_30109"]

pub mod quux {
    extern crate issue_30109_1 as bar;
    use self::bar::Bar;

    pub trait Foo {}

    //@ has issue_30109/quux/trait.Foo.html \
    //          '//a/@href' '../issue_30109_1/struct.Bar.html'
    impl Foo for Bar {}
}
