//! regression test for <https://github.com/rust-lang/rust/issues/17351>
//@ check-pass

trait Str { fn foo(&self) {} }
impl Str for str {}
impl<'a, S: ?Sized> Str for &'a S where S: Str {}

fn main() {
    let _: &dyn Str = &"x";
}
