//@ check-pass

struct Arg<'a: 'b, 'b, 'c> {
    field: *mut (&'a (), &'b (), &'c ()),
}
fn foo<'a, 'b, T: for<'c> FnOnce(Arg<'a, 'b, 'c>)>(_: T) {}

fn error<'a, 'b>() {
    foo::<'a, 'b>(|arg| {});
}
fn main() {}
