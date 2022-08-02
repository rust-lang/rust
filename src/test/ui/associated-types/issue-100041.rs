trait Wf {
    type Ty;
}

impl<T: ?Sized> Wf for T {
    type Ty = ();
}

const _: <Vec<str> as Wf>::Ty = ();
//~^ ERROR the size for values of type `str` cannot be known at compilation time

struct Foo {
    x: <Vec<str> as Wf>::Ty,
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}

fn foo(x: <Vec<str> as Wf>::Ty) {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time

fn bar() -> <Vec<str> as Wf>::Ty {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
