//@ known-bug: #133597

pub trait Foo2 {
    fn boxed<'a: 'a>() -> impl Sized + FnOnce<()>;
}

impl Foo2 for () {}


fn f() -> impl FnOnce<()> { || () }
fn main() { () = f(); }
