//@ known-bug: #153375
//@ aux-build: aux153375.rs
extern crate aux153375;
use aux153375::Request;

struct Bar<'ws>(&'ws ());

impl<'ws> Request for Bar<'ws> {
    type A<'a>
        = u8
    where
        Self: 'a;

    fn f(_: Self::A<'_>) -> impl Sized {}
}
