//@aux-build:foreign_blanket_impl.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@forbid-output: DoNotMentionThis
extern crate foreign_blanket_impl;
use foreign_blanket_impl::{DoNotMentionThis, ForeignType};

pub struct LocalType<T>(pub T);

#[diagnostic::do_not_recommend]
impl<T: DoNotMentionThis> Clone for LocalType<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}

fn main() {
    {
        let f = ForeignType(42_u8);
        let _ = Clone::clone(&f);
        //~^ERROR the trait bound `ForeignType<u8>: Clone` is not satisfied [E0277]

        let _ = f.clone();
        //~^ERROR no method named `clone` found for struct `ForeignType<T>` in the current scope [E0599]

    }

    {
        let l = LocalType(42_u8);
        let _ = Clone::clone(&l);
        //~^ERROR the trait bound `LocalType<u8>: Clone` is not satisfied [E0277]

        let _ = l.clone();
        //~^ERROR no method named `clone` found for struct `LocalType<T>` in the current scope [E0599]

    }
}
