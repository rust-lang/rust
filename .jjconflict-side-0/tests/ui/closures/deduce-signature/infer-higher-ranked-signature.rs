//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Foo {}
fn needs_foo<T>(_: T)
where
    Wrap<T>: Foo,
{
}

struct Wrap<T>(T);
impl<T> Foo for Wrap<T> where T: for<'a> Fn(&'a i32) {}

fn main() {
    needs_foo(|x| {
        x.to_string();
    });
}
