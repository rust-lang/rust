//! Regression test for <https://github.com/rust-lang/rust/issues/111852>.

//@ edition:2018
//@ check-pass

#![allow(unused)]

trait G: Send {
    type Gat<'l>: Send
    where
        Self: 'l;

    fn as_gat(&self) -> Self::Gat<'_>;
}

fn a(g: impl G) {
    let _: &dyn Send = &async move {
        let _gat = g.as_gat();
        async{}.await
    };
}

fn main() { }
