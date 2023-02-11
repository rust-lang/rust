// check-pass

trait Base<'f> {
    type Assoc;

    fn do_something(&self);
}

trait ForAnyLifetime: for<'f> Base<'f> {}

impl<T> ForAnyLifetime for T where T: for<'f> Base<'f> {}

trait CanBeDynamic: ForAnyLifetime + for<'f> Base<'f, Assoc = ()> {}

fn foo(a: &dyn CanBeDynamic) {
    a.do_something();
}

struct S;

impl<'a> Base<'a> for S {
    type Assoc = ();

    fn do_something(&self) {}
}

impl CanBeDynamic for S {}

fn main() {
    let s = S;
    foo(&s);
}
